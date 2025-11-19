#!/bin/bash
# ============================================================
#  Bash script: install_unZipro.sh
#  Install environment and dependencies for unZipro
# ============================================================
git clone https://github.com/Gabriel-Qin/unZipro.git
cd unZipro

conda create -n unZipro python=3.9
conda activate unZipro

pip install numpy pandas biotite requests
# pip install learn2learn ### Simply skip this given it might cause conflicts. We have upload a clean version of learn2learn to the GitHub repo.
# Install PyTorch (CUDA 12.4)
pip install torch==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124
# Note: PyTorch ≥ 2.0 is also supported. Please select the appropriate CUDA version for your GPU.
# FYI, please refer to: https://pytorch.org/get-started/locally/

echo "✅ unZipro installation complete. Environment: unZipro"