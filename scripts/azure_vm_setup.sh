#!/usr/bin/env bash
# ============================================================================
# Azure GPU VM Setup Script
# ============================================================================
# Sets up an Azure GPU VM for fine-tuning Llama-3.1-8B-Instruct with QLoRA.
#
# Usage:
#   1. Create an Azure VM (recommended: Standard_NC6s_v3 or NC24ads_A100_v4)
#   2. SSH into the VM
#   3. Clone your repo: git clone <repo-url> && cd MasterThesis_ACT_CAI-1
#   4. Run: bash scripts/azure_vm_setup.sh
#
# Tested on: Ubuntu 22.04 LTS with NVIDIA GPU
# ============================================================================

set -euo pipefail

echo "========================================"
echo " Azure GPU VM Setup"
echo "========================================"

# ------------------------------------------------------------------
# 1. System packages
# ------------------------------------------------------------------
echo "[1/6] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-pip python3-venv python3-dev \
    git git-lfs wget curl \
    build-essential cmake \
    > /dev/null 2>&1

# ------------------------------------------------------------------
# 2. NVIDIA driver check
# ------------------------------------------------------------------
echo "[2/6] Checking NVIDIA drivers..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. Installing CUDA toolkit..."
    # Azure GPU VMs usually come with drivers pre-installed.
    # If not, install the CUDA toolkit:
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update -qq
    sudo apt-get install -y -qq cuda-toolkit-12-4 > /dev/null 2>&1
    rm -f cuda-keyring_1.1-1_all.deb
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ------------------------------------------------------------------
# 3. Python virtual environment
# ------------------------------------------------------------------
echo "[3/6] Creating Python virtual environment..."
VENV_DIR="${HOME}/.venv_act_cai"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel -q

# ------------------------------------------------------------------
# 4. Install Python packages
# ------------------------------------------------------------------
echo "[4/6] Installing Python packages..."

# Core + training deps from requirements.txt
pip install -q \
    numpy pandas packaging \
    huggingface-hub datasets safetensors \
    openai jinja2 tqdm python-dotenv PyYAML \
    nltk scikit-learn scipy \
    accelerate bitsandbytes peft sentencepiece \
    transformers trl \
    wandb tensorboard \
    azure-ai-projects azure-identity azure-ai-ml \
    matplotlib seaborn

# Unsloth (install from source for latest GPU support)
echo "[4/6] Installing Unsloth..."
pip install -q "unsloth @ git+https://github.com/unslothai/unsloth.git"

# Flash attention (if Ampere+ GPU)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
if echo "${GPU_NAME}" | grep -qiE "A100|A10|H100|L40|RTX 40|RTX 30"; then
    echo "[4/6] Installing Flash Attention 2..."
    pip install -q flash-attn --no-build-isolation 2>/dev/null || \
        echo "  Flash Attention install failed (non-critical, will use default attention)"
fi

# ------------------------------------------------------------------
# 5. Environment variables
# ------------------------------------------------------------------
echo "[5/6] Setting up environment..."

ENV_FILE="$(pwd)/.env"
if [ ! -f "${ENV_FILE}" ]; then
    echo "Creating .env template..."
    cat > "${ENV_FILE}" << 'ENVEOF'
# Azure OpenAI (data generation + critique-revision)
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
OPENAI_API_VERSION=2025-04-01-preview

# Azure AI Foundry (Claude-Opus evaluation)
AZURE_AI_EVAL_ENDPOINT=https://<resource>.services.ai.azure.com/openai/deployments/claude-opus
AZURE_AI_EVAL_API_KEY=

# Critique-revision model deployment name
AZURE_AI_CRITIQUE_MODEL=o4-mini

# WandB
WANDB_API_KEY=
WANDB_ENTITY=alha8035-stockholm-university
WANDB_PROJECT=pilot_model0_sft

# HuggingFace (optional, for dataset access)
HF_TOKEN=
ENVEOF
    echo "  .env template created at ${ENV_FILE}"
    echo "  >>> IMPORTANT: Fill in your API keys before running notebooks! <<<"
else
    echo "  .env already exists, skipping template creation"
fi

# ------------------------------------------------------------------
# 6. NLTK data
# ------------------------------------------------------------------
echo "[6/6] Downloading NLTK data..."
python3 -c "
import nltk
for pkg in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger',
            'averaged_perceptron_tagger_eng', 'vader_lexicon']:
    nltk.download(pkg, quiet=True)
print('NLTK data downloaded')
"

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo ""
echo "========================================"
echo " Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Fill in API keys in .env"
echo "  2. Activate venv:  source ${VENV_DIR}/bin/activate"
echo "  3. Run notebooks:  cd notebooks && jupyter lab"
echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""
echo "Python: $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
