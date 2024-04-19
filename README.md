# OCRPY
This repository contains OCR algorithm implementations.
Supported models include:
|    Model    | Paper                            |
|:-----------:|:--------------------------------:|
|CRNN         | https://arxiv.org/abs/1507.05717 |
|...          | ....                             |

## Table of contents
- [Overview](#overview)
- [Installation](#installation)
- [Datasets](#datasets)
- [Usage](#Usage)
  - [Train](#train)
  - [Predict](#predict)
- [TODOs](#todos)
- [Result](#result)

## Installation 
1. Create venv:   `/opt/miniconda3/bin/python3 -m venv ocrpy3.8`
2. Activate venv: `source ocrpy3.8/bin/activate`
3. Install torch from whl: `pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113`
4. Install the rest of the requirements `pip install -r requirements.txt`
5. Install ocrpy `pip install -e .`

## Usage
Both training and testing configs can be found in the `configs` directory.

### Train
Training locally:
```
python ocrpy/train.py --config configs/train_mjsynth.py --name mjs-crnn-lr1e-4-epoch10 --lr 1e-4 --epoch 10 --amp true --batch 512
```
Training on slurm:
```
source submit_train.sh
```
Change config in 'train.sh'

### Predict
After training, arbitrary image can be predicted on.
```
python ocrpy/predict.py --image_path ... --weights_path ...
```

Input: <span align="center"><img src="figures/Available.png"/></span>

Output:

```text
Build CRNN model successfully.
Load CRNN model weights `./results/pretrained_models/CRNN-MJSynth-e9341ede.pth.tar` successfully.
``./figures/Available.png` -> `available`
```

## TODOs
[ ] Use `omegaconf` <br />
[ ] Integrate `torch-lightning` <br />
[ ] Add guide and results on trdg

## Result

Source of original paper results: [https://arxiv.org/pdf/1507.05717.pdf](https://arxiv.org/pdf/1507.05717.pdf).<br />
Our implementation differs in training details: image min-max normalization, `clip_grad_norm_=5`, Adam optimizer and cosine annealing lr schedule, batch size of 512 + beam search decoder with `beam_size=10`.<br />
General problem with MJSynth and SynthText training though is the underrepresentation of number characters. (14k ~ 7M in MJSynth).<br />
We therefore opt to synthesize our own data with trdg (TextRecognitionDataGenerator)
|    Model    |    IIIT5K    |IC13(alphanumeric)|  IC13(all) | MJSynth |
|:-----------:|:------------:|:----------------:|:----------:|:-------:|
| CRNN(paper) |     78.2     |       86.7       |    77.4    |   93.9  |
| CRNN(ours)  |   **84.7**   |        -         |  **80.1**  | **94.2**|

```bash
# Download `CRNN-Synth90k-e9341ede.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python predict.py --image_path ./figures/Available.png --weights_path ./results/pretrained_models/CRNN-MJSynth-e9341ede.pth.tar
```

## Credit

### An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition

_Baoguang Shi, Xiang Bai, Cong Yao_ <br>

**Abstract** <br>
Image-based sequence recognition has been a long-standing research topic in computer vision. In this paper, we
investigate the problem of scene text recognition, which is among the most important and challenging tasks in
image-based sequence recognition. A novel neural network architecture, which integrates feature extraction, sequence
modeling and transcription into a unified framework, is proposed. Compared with previous systems for scene text
recognition, the proposed architecture possesses four distinctive properties: (1) It is end-to-end trainable, in
contrast to most of the existing algorithms whose components are separately trained and tuned. (2) It naturally handles
sequences in arbitrary lengths, involving no character segmentation or horizontal scale normalization. (3) It is not
confined to any predefined lexicon and achieves remarkable performances in both lexicon-free and lexicon-based scene
text recognition tasks. (4) It generates an effective yet much smaller model, which is more practical for real-world
application scenarios. The experiments on standard benchmarks, including the IIIT-5K, Street View Text and ICDAR
datasets, demonstrate the superiority of the proposed algorithm over the prior arts. Moreover, the proposed algorithm
performs well in the task of image-based music score recognition, which evidently verifies the generality of it.

[[Paper]](https://arxiv.org/pdf/1507.05717) [[Code(Lua)]](https://github.com/bgshih/crnn)

```bibtex
@article{ShiBY17,
  author    = {Baoguang Shi and
               Xiang Bai and
               Cong Yao},
  title     = {An End-to-End Trainable Neural Network for Image-Based Sequence Recognition
               and Its Application to Scene Text Recognition},
  journal   = {{IEEE} Trans. Pattern Anal. Mach. Intell.},
  volume    = {39},
  number    = {11},
  pages     = {2298--2304},
  year      = {2017}
}
```