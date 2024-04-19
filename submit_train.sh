#sbatch train.sh --name mjsynth-crnn-lr1e-3-epoch5-fp --lr 1e-3 --epoch 5 --amp false
#sbatch train.sh --name mjsynth-crnn-lr1e-3-epoch5-hp --lr 1e-3 --epoch 5 --amp true
sbatch train.sh --name trdg4-crnn-lr5e-4-epoch5 --lr 5e-4 --epoch 5 --amp true --batch 64
