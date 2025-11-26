# Mamba Installation Guide

## Issue: Mamba requires CUDA

The `mamba-ssm` library requires NVIDIA CUDA to compile and cannot be installed on macOS or CPU-only machines. You'll need access to a machine with an NVIDIA GPU and CUDA installed.

## Solutions

### Option 1: Install on GPU Machine (Recommended)

When you have access to a GPU machine with CUDA:

```bash
# Install CUDA toolkit first (if not already installed)
# Check: nvcc --version

# Then install mamba-ssm
pip install mamba-ssm causal-conv1d
```

**Requirements:**
- CUDA 11.7 or higher
- PyTorch with CUDA support
- NVIDIA GPU (Ampere or newer recommended)

### Option 2: Use Google Colab (Free GPU)

```python
# In a Colab notebook with GPU runtime
!pip install mamba-ssm causal-conv1d
```

### Option 3: Use Cloud GPU (GCP, AWS, etc.)

Set up a VM with GPU support and install there.

### Option 4: Develop Without Mamba (Current Strategy)

For now, you can:
1. **Build the entire data pipeline** (works without GPU)
2. **Implement the Transformer baseline** (can work on CPU/MPS)
3. **Set up vision encoder and projection layers** (CPU/MPS compatible)
4. **Prepare the Mamba decoder architecture** (code without running)
5. **Move to GPU machine for Mamba training**

## Alternative: Pure PyTorch Mamba Implementation

If you can't access CUDA, you can implement Mamba from scratch in pure PyTorch (slower but works on CPU/MPS):

```bash
# Clone a pure PyTorch implementation
git clone https://github.com/johnma2006/mamba-minimal
```

This won't have the CUDA optimizations but will let you develop and test locally.

## Verification

After installing on GPU machine:

```python
import torch
from mamba_ssm import Mamba

# Test
batch_size, seq_len, d_model = 2, 64, 256
x = torch.randn(batch_size, seq_len, d_model).cuda()
model = Mamba(d_model=d_model).cuda()
output = model(x)
print(f"Success! Output shape: {output.shape}")
```

## Current Project Strategy

**Week 1-2 (Now):**
- Build data pipeline ✓
- Create preprocessing modules ✓
- Implement vision encoder ✓
- Test everything on CPU/MPS

**Week 2-3 (Need GPU):**
- Install mamba-ssm on GPU machine
- Implement Mamba decoder
- Train Mamba model

**Week 3-4:**
- Implement Transformer baseline
- Train and compare
