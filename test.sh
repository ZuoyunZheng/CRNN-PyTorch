#!/bin/bash
#SBATCH --job-name=mjsynth-crnn
#SBATCH --partition=main
#SBATCH -o /home/zuoyun.zheng/logs/%j.out
#SBATCH -e /home/zuoyun.zheng/logs/%j.err
#SBATCH --time=0:20:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=0

args=()
for i in "$@"; do
    args+=" $i"
done
/home/zuoyun.zheng/bin/python/crnn3.8/bin/python /home/zuoyun.zheng/gitlab/ocrpy/ocrpy/test.py $args;
