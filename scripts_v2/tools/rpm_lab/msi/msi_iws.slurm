#!/bin/bash
# One stage of an Interactive World Simulator training on MSI. Submit with msi_submit.py, never with plain sbatch: it
# pins the commit, sets the job name, account and log paths, and passes the run parameters:
#   RPM_PROJECT_DIR, RPM_BRANCH, RPM_COMMIT (full sha), RPM_COMMIT_FROM=msi_submit:<sha>,
#   RUN (run name), STAGE (1|2|3), DATASET (under data/iws/), LOAD_AE_RUN (previous stage's run, stages 2-3),
#   MAX_TIME (Lightning DD:HH:MM:SS, before the Slurm limit), TRAIN_ARGS_B64 (base64 JSON list of extra arguments),
#   MODE (train | eval), EVAL_RUNS (eval: ':'-separated runs whose newest checkpoints to validate with rollouts).
# Checkpoints are saved every 2000 steps (about 10 min on an H100); training stopped by MAX_TIME loses what came after
# the last one (Lightning does not save on a step-based schedule's stop).
# The training resumes from the run's newest checkpoint (full Lightning state), so a requeue or the next job of a
# chain continues the same run.
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail

: "${RPM_PROJECT_DIR:?}" "${RPM_BRANCH:?}" "${RUN:?}" "${STAGE:?}" "${DATASET:?}"
export RPM_COMMIT="${RPM_COMMIT:-}" RPM_COMMIT_FROM="${RPM_COMMIT_FROM:-}"
export MODE="${MODE:-train}" EVAL_RUNS="${EVAL_RUNS:-}"
export LOAD_AE_RUN="${LOAD_AE_RUN:-}" MAX_TIME="${MAX_TIME:-00:23:30:00}" TRAIN_ARGS_B64="${TRAIN_ARGS_B64:-W10=}"
export REPO="${RPM_PROJECT_DIR}/UWLab"
export WORKTREE="${RPM_PROJECT_DIR}/UWLab-run-${RUN}"
export VENV="${RPM_PROJECT_DIR}/venv-iws"
export DATA_DIR="${RPM_PROJECT_DIR}/data/iws/${DATASET}"
export RUNS="${RPM_PROJECT_DIR}/runs"
export RUN STAGE
SANDBOX="${RPM_PROJECT_DIR}/iws_sandbox"
IWS_SUBMODULE="third_party/interactive_world_sim"

if [[ ! ${RPM_COMMIT} =~ ^[0-9a-f]{40}$ ]]; then
    echo "FATAL: RPM_COMMIT is not a full commit sha ('${RPM_COMMIT}')"; exit 1
fi
if [[ ${RPM_COMMIT_FROM} != "msi_submit:${RPM_COMMIT}" ]]; then
    echo "FATAL: RPM_COMMIT was not set by msi_submit.py"; exit 1
fi
echo "job ${SLURM_JOB_ID:-?} run ${RUN} mode ${MODE} ${EVAL_RUNS} stage ${STAGE} dataset ${DATASET} commit ${RPM_COMMIT} on $(hostname)"

# One worktree per run, at exactly the pinned commit. All jobs share one repo: serialize on a lock file.
(
    flock -w 900 9 || { echo "FATAL: worktree lock timeout"; exit 1; }
    has() { git -C "${REPO}" cat-file -e "${RPM_COMMIT}^{commit}" 2>/dev/null; }
    has || git -C "${REPO}" fetch --quiet origin "${RPM_BRANCH}" || true
    has || { echo "FATAL: commit ${RPM_COMMIT} is not in ${REPO} (push it, or fetch on the login node)"; exit 1; }
    git -C "${REPO}" worktree prune
    # Read the list whole: piping it into grep -q under pipefail can SIGPIPE git and take the wrong branch.
    registered=$(git -C "${REPO}" worktree list --porcelain)
    if ! grep -qxF "worktree ${WORKTREE}" <<<"${registered}"; then
        rm -rf "${WORKTREE}"
        git -C "${REPO}" worktree add --detach "${WORKTREE}" "${RPM_COMMIT}"
    fi
    git -C "${WORKTREE}" checkout --quiet --force --detach "${RPM_COMMIT}"
    # The IWS submodule, cloned from the main checkout's copy: no network needed on the compute node.
    git -C "${WORKTREE}" -c protocol.file.allow=always \
        -c "submodule.${IWS_SUBMODULE}.url=${REPO}/${IWS_SUBMODULE}" \
        submodule update --init --quiet "${IWS_SUBMODULE}" || {
        echo "FATAL: IWS submodule commit missing; on the login node: git -C ${REPO} submodule update --init"; exit 1
    }
) 9>"${RPM_PROJECT_DIR}/.launcher_worktree.lock"

cd "${WORKTREE}"
if [ "$(git rev-parse HEAD)" != "${RPM_COMMIT}" ] || [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "FATAL: ${WORKTREE} is not exactly ${RPM_COMMIT}"; git status --short | head; exit 1
fi
shopt -s nullglob
caches=("${DATA_DIR}"/train/cache*.zarr.zip)
[ -f "${DATA_DIR}/manifest.json" ] && [ "${#caches[@]}" -gt 0 ] || {
    echo "FATAL: no IWS dataset at ${DATA_DIR} (push it with msi_sync.sh push-data)"; exit 1
}
mkdir -p "${RUNS}"
nvidia-smi

INNER_SCRIPT=$(mktemp "/tmp/iws_${RUN}_inner.XXXXXX.sh")
cat > "${INNER_SCRIPT}" <<'INNER'
#!/bin/bash
set -euo pipefail
shopt -s nullglob
export PATH="${VENV}/bin:${PATH}"
export PYTHONUNBUFFERED=1
cd "${WORKTREE}"
[ "$(git rev-parse HEAD)" = "${RPM_COMMIT}" ] || { echo "FATAL: wrong HEAD"; exit 1; }
newest_ckpt() {
    local newest="" ckpt
    for ckpt in "${RUNS}/$1"/checkpoints/*.ckpt; do
        [ -z "${newest}" ] || [ "${ckpt}" -nt "${newest}" ] && newest="${ckpt}"
    done
    echo "${newest}"
}
mapfile -t extra < <(python -c "import base64, json, os; print('\n'.join(json.loads(base64.b64decode(os.environ['TRAIN_ARGS_B64']))))")
if [ "${MODE}" = "eval" ]; then
    # rollouts of each run's newest checkpoint, under the recorded and under held actions
    IFS=: read -r -a eval_runs <<<"${EVAL_RUNS}"
    for run in "${eval_runs[@]}"; do
        ckpt=$(newest_ckpt "${run}")
        [ -n "${ckpt}" ] || { echo "FATAL: no checkpoint in ${RUNS}/${run}/checkpoints"; exit 1; }
        for actions in true hold; do
            args=(--name "${RUN}_${run}_${actions}" --data "${DATA_DIR}" --runs "${RUNS}" --workers 6
                  --eval "${ckpt}" --eval_actions "${actions}")
            for a in "${extra[@]}"; do [ -n "${a}" ] && args+=("${a}"); done
            echo "train_iws.py ${args[*]}"
            python scripts_v2/tools/rpm_lab/world_model/train_iws.py "${args[@]}"
        done
    done
    exit 0
fi
args=(--stage "${STAGE}" --name "${RUN}" --data "${DATA_DIR}" --runs "${RUNS}"
      --resume --max_time "${MAX_TIME}" --keep_last 2 --ckpt_every 2000 --workers 6)
if [ -n "${LOAD_AE_RUN}" ]; then
    # the previous stage's newest checkpoint
    newest=$(newest_ckpt "${LOAD_AE_RUN}")
    [ -n "${newest}" ] || { echo "FATAL: no checkpoint in ${RUNS}/${LOAD_AE_RUN}/checkpoints"; exit 1; }
    args+=(--load_ae "${newest}")
fi
for a in "${extra[@]}"; do [ -n "${a}" ] && args+=("${a}"); done
echo "train_iws.py ${args[*]}"
python scripts_v2/tools/rpm_lab/world_model/train_iws.py "${args[@]}"
INNER
chmod +x "${INNER_SCRIPT}"
apptainer exec --nv --bind "${RPM_PROJECT_DIR}:${RPM_PROJECT_DIR}" "${SANDBOX}" bash "${INNER_SCRIPT}"
rm -f "${INNER_SCRIPT}"
