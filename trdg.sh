#!/bin/bash
#SBATCH --job-name=trdg
#SBATCH --partition=main
#SBATCH -o /home/zuoyun.zheng/logs/%j.out
#SBATCH -e /home/zuoyun.zheng/logs/%j.err
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=0

# split into 10k image chunks
args=()
for (( i=1; i<=$#; i++ )); do
    args+=" ${!i}"
    #if [ "${!i}" = "-c" ]; then
    #    j=$((i+1))
    #    count=${!j}
    #fi
done

source /home/zuoyun.zheng/bin/python/crnn3.8/bin/activate
trdg -rs -let -num -sym -f 64 -t 4 -k 5 -rk -obb 1 -d 4 -do 2 -al 3 -tc "#000000,#FFFFFF" -cs 5 -m 2,2,2,2 -rm -fi -fd ~/github/TextRecognitionDataGenerator/trdg/fonts/custom_fonts_2 -id ~/data/lvis_v1_val/val2017/ -stw 3 -stf "#000000,#FFFFFF" -rst $args;
# debug
#trdg -rs -let -num -f 64 -obb 1 -d 0 -al 3 -tc "#000000,#FFFFFF" -cs 5 -rm -fd ~/github/TextRecognitionDataGenerator/trdg/fonts/myfonts -id ~/data/lvis_v1_val/val2017/ -stw 3 -stf "#000000,#FFFFFF" -rst $args;
