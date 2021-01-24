#!/bin/bash

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license

cd "$(dirname "$0")"
cd ..

# Pytorch standard implementation
python3 -m torch.distributed.launch --nproc_per_node=1 model_train_kitti.py --dataset KITTI --use-sigmoid --img-channels 2 --img-height 128 --img-width 128 --kernel-size 5 --model convlstm --batch-size 1 --learning-rate 1e-3 --valid-samples 448 --num-epochs 500 --ssr-decay-ratio 4e-3 --valid-data-file 'test' # --use-amp --use-checkpointing
