#!/bin/bash
#SBATCH --job-name=mjsynth-crnn-lr1e-3-epoch5
#SBATCH --partition=main
#SBATCH -o /home/zuoyun.zheng/logs/%j.out
#SBATCH -e /home/zuoyun.zheng/logs/%j.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1

args=()
for i in "$@"; do
    args+=" $i"
done
#/home/zuoyun.zheng/bin/python/crnn3.8/bin/python /home/zuoyun.zheng/gitlab/ocrpy/ocrpy/train.py --config /home/zuoyun.zheng/gitlab/ocrpy/configs/train_mjsynth.py $args;
/home/zuoyun.zheng/bin/python/crnn3.8/bin/python /home/zuoyun.zheng/gitlab/ocrpy/ocrpy/train.py --config /home/zuoyun.zheng/gitlab/ocrpy/configs/train_trdg.py $args;
