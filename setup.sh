#!/bin/bash
# NVIDIA Cosmos Reason 8B - Quick Installation Script

set -e  # Exit on error

echo "=============================================="
echo "NVIDIA Cosmos Reason 8B Quick Setup"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${RED}This script is designed for Linux. For other OS, please follow the manual installation in COSMOS_SETUP_GUIDE.md${NC}"
    exit 1
fi

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not found. Please install NVIDIA drivers first.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Check GPU memory
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$GPU_MEM" -lt 20000 ]; then
    echo -e "${YELLOW}WARNING: Your GPU has less than 20GB memory. Consider using the 2B model instead of 8B.${NC}"
fi

# Install system dependencies
echo ""
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y curl ffmpeg git git-lfs python3-pip python3-venv

# Create virtual environment
echo ""
echo "Creating Python virtual environment..."
VENV_DIR="cosmos_env"

if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Skipping creation.${NC}"
else
    python3 -m venv $VENV_DIR
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch
echo ""
echo "Installing PyTorch with CUDA support..."
echo "This may take several minutes..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify CUDA
echo ""
echo "Verifying PyTorch CUDA installation..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Install dependencies
echo ""
echo "Installing Cosmos Reason dependencies..."
pip install transformers>=4.57.0
pip install accelerate
pip install qwen-vl-utils
pip install pillow
pip install opencv-python
pip install huggingface-hub

# Check for Hugging Face token
echo ""
echo "=============================================="
echo "Hugging Face Login Required"
echo "=============================================="
echo "You need a Hugging Face account to download the models."
echo "If you don't have one, create one at: https://huggingface.co/join"
echo "Then create a token at: https://huggingface.co/settings/tokens"
echo ""
read -p "Press Enter to login to Hugging Face..."

$VENV_DIR/bin/hf auth login

# Download demo script if not exists
if [ ! -f "cosmos_reason_demo.py" ]; then
    echo ""
    echo -e "${YELLOW}Demo script not found in current directory.${NC}"
    echo "Please ensure cosmos_reason_demo.py is in the same directory as this script."
fi

# Test installation
echo ""
echo "=============================================="
echo "Testing Installation"
echo "=============================================="
echo "Running a quick test..."

python3 -c "
import torch
from transformers import AutoProcessor
print('✓ PyTorch version:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ CUDA device:', torch.cuda.get_device_name(0))
    print('✓ GPU memory:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), 'GB')
print('✓ Transformers imported successfully')
print('✓ Processor available')
"

echo ""
echo "=============================================="
echo "Installation Complete!"
echo "=============================================="
echo ""
echo -e "${GREEN}Setup successful!${NC}"
echo ""
echo "To activate the environment in the future:"
echo "  source cosmos_env/bin/activate"
echo ""
echo "To run the demo:"
echo "  python cosmos_reason_demo.py --video /path/to/your/video.mp4"
echo ""
echo "For more examples, see COSMOS_SETUP_GUIDE.md"
echo ""
echo -e "${YELLOW}Note: The first time you run the script, it will download the model (~16GB for 8B model).${NC}"
echo "This may take 10-30 minutes depending on your internet connection."
echo ""