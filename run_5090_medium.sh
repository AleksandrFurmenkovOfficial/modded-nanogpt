#!/usr/bin/env bash
set -euo pipefail

ATTN_BACKEND=flash_attn GPU_PROFILE=5090 GRAD_ACCUM_STEPS=8 \
TRAIN_BS_SCHEDULE=18432,36864,55296 TRAIN_BS_EXTENSION=55296 \
torchrun --standalone --nproc_per_node=1 train_gpt_medium.py
