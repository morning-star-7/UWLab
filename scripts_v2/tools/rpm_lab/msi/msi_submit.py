# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Submit one Interactive World Simulator training stage to MSI (Agate) at one pushed commit.

The job trains exactly the commit at the tip of ``--branch`` on origin: push first. This tool reads the launcher
(``msi_iws.slurm``) from that commit, checks it (non-preemptible ``msigpu``, commit pin), checks over the open ssh
ControlMaster connection that the sandbox, the venv and the dataset exist on MSI and that no job of the same run is
queued, then pipes the launcher into ``sbatch`` with the commit pinned. ``--chain N`` submits N jobs, each starting
after the previous one ends (``afterany``) and resuming the run from its newest checkpoint: a stage longer than the
24 h ``msigpu`` limit. Each submission is appended to the job ledger (``LEDGER`` in ``msi.env``, or ``--ledger``).

    python scripts_v2/tools/rpm_lab/msi/msi_submit.py --run cf_ae --stage 1 --dataset cook_fish_r40_s500
    python scripts_v2/tools/rpm_lab/msi/msi_submit.py --run cf_dyn --stage 2 --dataset cook_fish_r40_s500 \\
        --load_ae_run cf_ae --chain 2 --dependency <stage-1 job id>
"""

import argparse
import base64
import datetime
import json
import os
import re
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCHER = "scripts_v2/tools/rpm_lab/msi/msi_iws.slurm"
NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def read_env(path: str) -> dict:
    values = {}
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, **kwargs)


def fail(message: str):
    sys.exit(f"refused: {message}")


def remote(host: str, command: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, command], input=stdin)


def hours_minus(time_limit: str, minutes: int) -> str:
    """Lightning max_time (DD:HH:MM:SS) ``minutes`` before a Slurm HH:MM:SS limit."""
    h, m, s = (int(x) for x in time_limit.split(":"))
    total = max(60, h * 3600 + m * 60 + s - minutes * 60)
    return f"{total // 86400:02d}:{total % 86400 // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Run name (job name, runs/<run>, worktree UWLab-run-<run>).")
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--dataset", required=True, help="Dataset directory name under data/iws/ on MSI.")
    parser.add_argument("--load_ae_run", default="", help="Previous stage's run (stages 2 and 3).")
    parser.add_argument("--branch", default=None, help="Branch to train (default: the current branch).")
    parser.add_argument("--time", default="24:00:00", help="Slurm time limit, HH:MM:SS (msigpu max 24:00:00).")
    parser.add_argument("--gpu", default="a100|h100", help="Node constraint for the GPU type.")
    parser.add_argument("--chain", type=int, default=1, help="Jobs in a resume chain.")
    parser.add_argument("--dependency", default=None, help="Job id the first job waits for (afterok).")
    parser.add_argument("--env", default=os.path.join(HERE, "msi.env"))
    parser.add_argument("--ledger", default=None, help="Job ledger (JSON lines); default: LEDGER in msi.env.")
    parser.add_argument("--dry_run", action="store_true", help="Run the checks and print the sbatch command only.")
    parser.add_argument("train_args", nargs="*", help="Extra train_iws.py arguments (after --), e.g. --batch_size 4.")
    args = parser.parse_args()

    cfg = read_env(args.env)
    host, account, project = cfg.get("MSI_HOST", "msi"), cfg.get("MSI_ACCOUNT"), cfg.get("MSI_PROJECT_DIR")
    ledger = args.ledger or cfg.get("LEDGER") or os.path.join(HERE, "msi_jobs.jsonl")
    if not account or not project:
        fail(f"set MSI_ACCOUNT and MSI_PROJECT_DIR in {args.env}")
    for value in (args.run, args.dataset, *([args.load_ae_run] if args.load_ae_run else [])):
        if not NAME.match(value):
            fail(f"{value!r}: names may only use letters, digits, '_', '.', '-'")
    if args.stage > 1 and not args.load_ae_run:
        fail(f"stage {args.stage} needs --load_ae_run")
    if args.time > "24:00:00" or not re.match(r"^\d\d:\d\d:\d\d$", args.time):
        fail("--time must be HH:MM:SS within msigpu's 24:00:00")

    # 1. the commit: the branch tip on origin (MSI fetches it from there)
    branch = args.branch or run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    if run(["git", "fetch", "--quiet", "origin", branch]).returncode != 0:
        fail(f"cannot fetch origin/{branch}")
    sha = run(["git", "rev-parse", f"origin/{branch}"]).stdout.strip()
    if run(["git", "rev-parse", "HEAD"]).stdout.strip() != sha:
        print(f"note: local HEAD is not origin/{branch}; the job trains origin/{branch} = {sha[:12]}")

    # 2. the launcher as committed at that sha
    text = run(["git", "show", f"{sha}:{LAUNCHER}"]).stdout
    if not text:
        fail(f"{LAUNCHER} is not in commit {sha[:12]} (push it first)")
    if "#SBATCH --partition=msigpu" not in text or "preempt" in text.lower():
        fail("the launcher must train on msigpu and never on a preemptible partition")
    if "msi_submit:${RPM_COMMIT}" not in text:
        fail("the launcher lacks the commit pin")

    # 3. MSI side, over the open ControlMaster connection
    if run(["ssh", "-O", "check", host]).returncode != 0:
        fail(f"no open ssh connection to {host}: open it by hand first (ssh -fN {host})")
    checks = (
        f"test -x {project}/venv-iws/bin/python && test -d {project}/iws_sandbox && test -d {project}/UWLab/.git"
        f" || echo MISSING_SETUP; test -f {project}/data/iws/{args.dataset}/manifest.json || echo MISSING_DATA;"
        f" squeue -h -u $USER -n {args.run} -o %i"
    )
    out = remote(host, checks)
    if out.returncode != 0:
        fail(f"ssh check failed: {out.stderr.strip()}")
    lines = out.stdout.split()
    if "MISSING_SETUP" in lines:
        fail(f"no sandbox / venv / repo under {project}: run setup_msi.sh on MSI first")
    if "MISSING_DATA" in lines:
        fail(f"no dataset {args.dataset} on MSI: msi_sync.sh push-data <local dataset dir>")
    queued = [x for x in lines if x.isdigit()]
    if queued and not args.dependency:
        fail(f"run {args.run} already has queued/running jobs {queued}; chain after them with --dependency")

    # 4. submit (a chain resumes from the newest checkpoint each time)
    exports = {
        "RPM_PROJECT_DIR": project,
        "RPM_BRANCH": branch,
        "RPM_COMMIT": sha,
        "RPM_COMMIT_FROM": f"msi_submit:{sha}",
        "RUN": args.run,
        "STAGE": str(args.stage),
        "DATASET": args.dataset,
        "LOAD_AE_RUN": args.load_ae_run,
        "MAX_TIME": hours_minus(args.time, 30),
        "TRAIN_ARGS_B64": base64.b64encode(json.dumps(args.train_args).encode()).decode(),
    }
    export = "ALL," + ",".join(f"{k}={v}" for k, v in exports.items())
    dependency = f"afterok:{args.dependency}" if args.dependency else None
    jobs = []
    for i in range(args.chain):
        sbatch = [
            "sbatch",
            f"--job-name={args.run}",
            f"--account={account}",
            f"--chdir={project}",
            f"--output={project}/logs/{args.run}_%j.out",
            f"--error={project}/logs/{args.run}_%j.err",
            f"--time={args.time}",
            f"--constraint={args.gpu}",
            f"--export={export}",
            *([f"--dependency={dependency}"] if dependency else []),
        ]
        command = f"mkdir -p {project}/logs && cd {project} && " + " ".join(shlex.quote(a) for a in sbatch)
        print(f"[{i + 1}/{args.chain}] ssh {host} {command}  < {LAUNCHER}@{sha[:12]}")
        if args.dry_run:
            dependency = "afterany:<previous>"
            continue
        out = remote(host, command, stdin=text)
        match = re.search(r"Submitted batch job (\d+)", out.stdout)
        if out.returncode != 0 or not match:
            fail(f"sbatch failed: {out.stdout.strip()} {out.stderr.strip()}")
        job = match.group(1)
        jobs.append(job)
        dependency = f"afterany:{job}"
        os.makedirs(os.path.dirname(os.path.abspath(ledger)), exist_ok=True)
        with open(ledger, "a") as f:
            record = {
                "time": datetime.datetime.now().isoformat(timespec="seconds"),
                "job": job,
                "run": args.run,
                "stage": args.stage,
                "dataset": args.dataset,
                "load_ae_run": args.load_ae_run,
                "commit": sha,
                "branch": branch,
                "time_limit": args.time,
                "chain": f"{i + 1}/{args.chain}",
                "train_args": args.train_args,
            }
            f.write(json.dumps(record) + "\n")
        print(f"submitted job {job}")
    if jobs:
        print(f"logs: {project}/logs/{args.run}_<job>.out   status: ssh {host} 'squeue -u $USER'")


if __name__ == "__main__":
    main()
