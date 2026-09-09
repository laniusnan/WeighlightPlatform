#!/usr/bin/env bash
# Create an isolated Python 3.11 ML runtime for decoder work (engine=nncf,
# optimum export probes) with the VERIFIED combination:
#   python 3.11 + torch 2.9.1(+cpu) + transformers 4.49 + tokenizers<0.22
# This stack is what made GPT-family decoder trace + NNCF INT4 succeed
# (py3.14 + transformers 4.57.x fails with TorchScript trace errors).
#
# Usage: bash scripts/setup-ml-py311.sh [venv_name]   (default: .venv311)
set -euo pipefail
cd "$(dirname "$0")/.."

VENV_NAME="${1:-.venv311}"
echo "==> Creating ${VENV_NAME} (python 3.11) ..."
uv venv --python 3.11 "${VENV_NAME}"

echo "==> Installing torch 2.9.1 (CPU wheels) ..."
uv pip install --python "${VENV_NAME}/bin/python" torch==2.9.1 \
  --index-url https://download.pytorch.org/whl/cpu

echo "==> Installing ML + platform deps ..."
uv pip install --python "${VENV_NAME}/bin/python" \
  pydantic pydantic-settings sqlmodel "psycopg[binary]" sqlalchemy \
  pyjwt "pwdlib[bcrypt]" argon2-cffi email-validator emails jinja2 httpx alembic \
  transformers==4.49.0 "tokenizers<0.22" safetensors numpy \
  openvino nncf optimum optimum-intel onnx onnxsim onnxruntime

echo "==> Done: ${VENV_NAME}"
echo "Verified decoder E2E:"
echo "  .venv311/bin/python scripts/e2e-nncf-int4.py            # generic-model INT4"
echo "  # then any engine=nncf path / NncfChatModel runtime test"
