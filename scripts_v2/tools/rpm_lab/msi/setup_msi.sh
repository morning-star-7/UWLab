#!/bin/bash
# One-time (idempotent) setup of the RPM Lab world-model training on MSI. Runs on the MSI login node, which has network
# access; from the workstation:
#
#   ssh msi "bash -s -- <MSI_PROJECT_DIR> <branch>" < scripts_v2/tools/rpm_lab/msi/setup_msi.sh
#
# 1. clones the repo (public fork, https) and the IWS submodule into <MSI_PROJECT_DIR>/UWLab;
# 2. builds the Apptainer sandbox <MSI_PROJECT_DIR>/iws_sandbox (thin: CUDA 12.6 devel, Python 3.11, uv);
# 3. creates the uv venv <MSI_PROJECT_DIR>/venv-iws through the sandbox and installs the IWS training deps;
# 4. imports the training stack inside the sandbox as a check.
# Re-running skips finished steps; delete iws_sandbox/ or venv-iws/ to redo them. DEF=<file> builds the sandbox from
# another copy of rpm_lab_iws.def (e.g. one copied over before it is pushed).
set -euo pipefail

PROJECT_DIR=${1:?usage: setup_msi.sh <MSI_PROJECT_DIR> [branch]}
BRANCH=${2:-hanchen/digit_cousin}
REPO_URL=${REPO_URL:-https://github.com/morning-star-7/UWLab.git}
REPO="${PROJECT_DIR}/UWLab"
SANDBOX="${PROJECT_DIR}/iws_sandbox"
VENV="${PROJECT_DIR}/venv-iws"
WM="scripts_v2/tools/rpm_lab/world_model"
DEF=${DEF:-${REPO}/scripts_v2/tools/rpm_lab/msi/rpm_lab_iws.def}
# image layers and build scratch on project storage, not the login node's /tmp
export APPTAINER_CACHEDIR="${PROJECT_DIR}/.apptainer_cache" APPTAINER_TMPDIR="${PROJECT_DIR}/.apptainer_tmp"

mkdir -p "${PROJECT_DIR}"/{data/iws,runs,logs,.uv_cache,.apptainer_cache,.apptainer_tmp}
command -v apptainer >/dev/null || module load apptainer 2>/dev/null || true
command -v apptainer >/dev/null || { echo "FATAL: apptainer not found"; exit 1; }

echo "== repo"
if [ ! -d "${REPO}/.git" ]; then
    git clone --branch "${BRANCH}" "${REPO_URL}" "${REPO}"
fi
git -C "${REPO}" fetch --quiet origin "${BRANCH}"
git -C "${REPO}" checkout --quiet "${BRANCH}"
git -C "${REPO}" merge --quiet --ff-only "origin/${BRANCH}"
git -C "${REPO}" submodule update --init third_party/interactive_world_sim
git -C "${REPO}" log --oneline -1

echo "== sandbox"
if [ ! -x "${SANDBOX}/usr/local/bin/uv" ]; then  # a failed build can leave a partial directory
    rm -rf "${SANDBOX}"
    apptainer build --sandbox "${SANDBOX}" "${DEF}" || {
        rm -rf "${SANDBOX}"
        apptainer build --sandbox --fakeroot "${SANDBOX}" "${DEF}"
    }
fi

echo "== venv"
run_in_sandbox() {
    apptainer exec --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
        --env "UV_CACHE_DIR=${PROJECT_DIR}/.uv_cache" "${SANDBOX}" bash -c "$1"
}
if [ ! -x "${VENV}/bin/python" ]; then
    run_in_sandbox "uv venv --python /usr/bin/python3.11 '${VENV}'"
fi
# VIRTUAL_ENV, not --python <venv>/bin/python: that python links to the read-only system prefix in the sandbox.
run_in_sandbox "cd '${REPO}' && VIRTUAL_ENV='${VENV}' uv pip install -r ${WM}/requirements.txt \
    --build-constraint ${WM}/build_constraints.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match"

echo "== check"
run_in_sandbox "cd '${REPO}/third_party/interactive_world_sim' && '${VENV}/bin/python' -c \"
import torch, lightning, zarr, hydra
import interactive_world_sim.experiments
from interactive_world_sim.utils.imagecodecs_numcodecs import register_codecs
register_codecs()
print('torch', torch.__version__, 'lightning', lightning.__version__, 'zarr', zarr.__version__, ': IWS imports ok')
\""
echo "Setup done: ${PROJECT_DIR}"
