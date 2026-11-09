#!/bin/bash
# Move world-model data between this workstation and MSI over the open ssh ControlMaster connection (msi.env).
#
#   msi_sync.sh push-data <local dataset dir>   # build_iws_dataset.py output -> MSI data/iws/<name>/
#   msi_sync.sh pull-run <run>                  # MSI runs/<run>/ -> LOCAL_WM_DIR/runs_msi/<run>/ (checkpoints, config,
#                                               #   logs; not the wandb directory), then checks sha256 of each file
#   msi_sync.sh status                          # your MSI jobs and the newest log lines of each running one
#   msi_sync.sh wandb                           # mirror the runs' offline wandb logs and keep a live uploader per run
#                                               #   (this machine's wandb login -> WANDB_ENTITY/WANDB_PROJECT): runs show
#                                               #   as running on wandb.ai and finish when the job does
#   msi_sync.sh wandb-watch [minutes]           # repeat `wandb` (default every 5 min) while jobs are queued/running on
#                                               #   MSI; run it detached (setsid nohup ... &)
#
# rsync -a keeps modification times.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
source "${HERE}/msi.env"
: "${MSI_PROJECT_DIR:?set MSI_PROJECT_DIR in msi.env}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20)
ssh -O check "${MSI_HOST}" 2>/dev/null || { echo "no open ssh connection: ssh -fN ${MSI_HOST} first"; exit 2; }

case "${1:-}" in
push-data)
    src=${2:?usage: msi_sync.sh push-data <local dataset dir>}
    [ -f "${src}/manifest.json" ] || { echo "${src} is not a build_iws_dataset.py output"; exit 1; }
    name=$(basename "$(realpath "${src}")")
    "${SSH[@]}" "${MSI_HOST}" "mkdir -p ${MSI_PROJECT_DIR}/data/iws/${name}"
    rsync -a --info=progress2 -e "${SSH[*]}" "${src}/" "${MSI_HOST}:${MSI_PROJECT_DIR}/data/iws/${name}/"
    echo "pushed ${src} -> ${MSI_HOST}:${MSI_PROJECT_DIR}/data/iws/${name}"
    ;;
pull-run)
    run=${2:?usage: msi_sync.sh pull-run <run>}
    dest="${LOCAL_WM_DIR}/runs_msi/${run}"
    mkdir -p "${dest}"
    rsync -a --info=progress2 -e "${SSH[*]}" --exclude wandb/ \
        "${MSI_HOST}:${MSI_PROJECT_DIR}/runs/${run}/" "${dest}/"
    (cd "${dest}" && find . -type f ! -name 'SHA256SUMS*' -print0 | sort -z | xargs -0 sha256sum) > "${dest}/SHA256SUMS.local"
    "${SSH[@]}" "${MSI_HOST}" "cd ${MSI_PROJECT_DIR}/runs/${run} && find . -path ./wandb -prune -o -type f -print0 \
        | sort -z | xargs -0 sha256sum" > "${dest}/SHA256SUMS"
    if diff -q "${dest}/SHA256SUMS" "${dest}/SHA256SUMS.local" >/dev/null; then
        echo "pulled ${run} -> ${dest} ($(wc -l < "${dest}/SHA256SUMS") files, sha256 match)"
    else
        echo "WARNING: sha256 differ (a job may still be writing):"; diff "${dest}/SHA256SUMS" "${dest}/SHA256SUMS.local" | head
    fi
    rm -f "${dest}/SHA256SUMS.local"
    ;;
status)
    "${SSH[@]}" "${MSI_HOST}" "squeue -u \$USER -o '%.10i %.20j %.9T %.11M %.11l %R'; \
        for j in \$(squeue -h -u \$USER -t RUNNING -o '%i:%j'); do \
            id=\${j%%:*}; name=\${j#*:}; echo \"== \${name} (\${id})\"; \
            tail -n 3 ${MSI_PROJECT_DIR}/logs/\${name}_\${id}.out 2>/dev/null | tr '\r' '\n' | tail -n 2; done"
    ;;
wandb)
    # One pass: mirror the runs' wandb directories (the append-only .wandb logs grow in place, so a live syncer can
    # tail them) and keep one `wandb beta sync --live` per offline run until it uploads the run's end.
    mirror="${LOCAL_WM_DIR}/wandb_mirror"
    mkdir -p "${mirror}"
    for run in $("${SSH[@]}" "${MSI_HOST}" "ls ${MSI_PROJECT_DIR}/runs"); do
        "${SSH[@]}" "${MSI_HOST}" "test -d ${MSI_PROJECT_DIR}/runs/${run}/wandb" || continue
        src="${MSI_HOST}:${MSI_PROJECT_DIR}/runs/${run}/wandb/"
        rsync -a -e "${SSH[*]}" --exclude '*.wandb' "${src}" "${mirror}/${run}/"
        rsync -a --append -e "${SSH[*]}" --include '*/' --include '*.wandb' --exclude '*' "${src}" "${mirror}/${run}/"
    done
    shopt -s nullglob
    for log in "${mirror}"/*/offline-run-*/run-*.wandb; do
        d=$(dirname "${log}")
        [ -f "${d}/.live_sync.done" ] && continue
        if [ -f "${d}/.live_sync.pid" ] && kill -0 "$(cat "${d}/.live_sync.pid")" 2>/dev/null; then continue; fi
        if [ -f "${d}/.live_sync.exit" ]; then
            if [ "$(cat "${d}/.live_sync.exit")" = 0 ]; then touch "${d}/.live_sync.done"; continue; fi
            echo "$(date '+%F %T') live sync of ${d} exited $(cat "${d}/.live_sync.exit"): restarting"
        fi
        rm -f "${d}/.live_sync.exit"
        setsid nohup bash -c "'${WANDB_PYTHON}' -m wandb beta sync --live -e '${WANDB_ENTITY}' -p '${WANDB_PROJECT}' \
            '${log}' > '${d}/live_sync.log' 2>&1; echo \$? > '${d}/.live_sync.exit'" < /dev/null > /dev/null 2>&1 &
        echo $! > "${d}/.live_sync.pid"
        echo "$(date '+%F %T') live sync started: ${d}"
    done
    echo "$(date '+%F %T') mirrored; runs at https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
    ;;
wandb-watch)
    minutes=${2:-5}
    fails=0
    while true; do
        "$0" wandb || true
        if queued=$("${SSH[@]}" "${MSI_HOST}" "squeue -h -u \$USER -o %i" 2>/dev/null); then
            fails=0
            [ -n "${queued}" ] || break
        else  # an ssh hiccup is not "no jobs": retry, give up after a day
            fails=$((fails + 1))
            echo "$(date '+%F %T') MSI unreachable (${fails}); is the ssh connection still open? (ssh -fN ${MSI_HOST})"
            [ "${fails}" -lt $((24 * 60 / minutes)) ] || { echo "MSI unreachable for a day: stopping"; break; }
        fi
        sleep $((minutes * 60))
    done
    # No jobs left: last mirror pass, let the live syncers upload the runs' ends, then stop any that hang (a run
    # whose job crashed never gets an end record).
    "$0" wandb || true
    sleep 1800
    for pidfile in "${LOCAL_WM_DIR}"/wandb_mirror/*/offline-run-*/.live_sync.pid; do
        kill "$(cat "${pidfile}")" 2>/dev/null && echo "stopped a hanging live sync: $(dirname "${pidfile}")"
    done
    echo "no MSI jobs left: stopped"
    ;;
*)
    sed -n '2,14p' "$0"; exit 1
    ;;
esac
