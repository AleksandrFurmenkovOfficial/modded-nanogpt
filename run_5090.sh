#!/usr/bin/env bash
set -euo pipefail

ATTN_BACKEND=flash_attn GPU_PROFILE=5090 GRAD_ACCUM_STEPS=4 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
