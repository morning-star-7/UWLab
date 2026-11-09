#!/bin/bash
# Move world-model data between this workstation and MSI over the open ssh ControlMaster connection (msi.env).
#
#   msi_sync.sh push-data <local dataset dir>   # build_iws_dataset.py output -> MSI data/iws/<name>/
#   msi_sync.sh pull-run <run>                  # MSI runs/<run>/ -> LOCAL_WM_DIR/runs_msi/<run>/ (checkpoints, config,
#                                               #   logs; not the wandb directory), then checks sha256 of each file
#   msi_sync.sh status                          # your MSI jobs and the newest log lines of each running one
#   msi_sync.sh wandb                           # pull the runs' offline wandb logs, upload them with this machine's
#                                               #   wandb login to WANDB_ENTITY/WANDB_PROJECT (a grown run re-syncs)
#   msi_sync.sh wandb-watch [minutes]           # repeat `wandb` (default every 15 min) while jobs are queued/running
#                                               #   on MSI, then once more; run it detached (setsid nohup ... &)
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
    mirror="${LOCAL_WM_DIR}/wandb_mirror"
    mkdir -p "${mirror}"
    for run in $("${SSH[@]}" "${MSI_HOST}" "ls ${MSI_PROJECT_DIR}/runs"); do
        "${SSH[@]}" "${MSI_HOST}" "test -d ${MSI_PROJECT_DIR}/runs/${run}/wandb" || continue
        rsync -a -e "${SSH[*]}" "${MSI_HOST}:${MSI_PROJECT_DIR}/runs/${run}/wandb/" "${mirror}/${run}/"
    done
    shopt -s nullglob
    for d in "${mirror}"/*/offline-run-*; do
        log=("${d}"/run-*.wandb)
        [ "${#log[@]}" -gt 0 ] || continue
        # re-sync a run whose log grew since its last upload (wandb keeps only the new steps)
        if [ -f "${log[0]}.synced" ] && [ "${log[0]}" -nt "${log[0]}.synced" ]; then rm -f "${log[0]}.synced"; fi
        [ -f "${log[0]}.synced" ] && continue
        "${WANDB_PYTHON}" -m wandb sync --entity "${WANDB_ENTITY}" --project "${WANDB_PROJECT}" "${d}" 2>&1 | grep -v "incomplete record" | tail -n 1
    done
    echo "$(date '+%F %T') synced to https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
    ;;
wandb-watch)
    minutes=${2:-15}
    fails=0
    while true; do
        "$0" wandb || true
        if queued=$("${SSH[@]}" "${MSI_HOST}" "squeue -h -u \$USER -o %i" 2>/dev/null); then
            fails=0
            [ -n "${queued}" ] || { "$0" wandb || true; echo "no MSI jobs left: stopping"; break; }
        else  # an ssh hiccup is not "no jobs": retry, give up after a day
            fails=$((fails + 1))
            echo "$(date '+%F %T') MSI unreachable (${fails}); is the ssh connection still open? (ssh -fN ${MSI_HOST})"
            [ "${fails}" -lt $((24 * 60 / minutes)) ] || { echo "MSI unreachable for a day: stopping"; break; }
        fi
        sleep $((minutes * 60))
    done
    ;;
*)
    sed -n '2,14p' "$0"; exit 1
    ;;
esac
