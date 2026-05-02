<p align="center">
  <h1 align="center">Controllable Texture Tiling with Transformed RoPE-Enhanced Diffusion Models</h1>
  <p align="center">
    <a href="https://scholars.cityu.edu.hk/en/persons/jrhuang8/"><strong>Junrong HUANG</strong></a>
    ·
    <a href="https://scholars.cityu.edu.hk/en/persons/zzhang452/"><strong>Zhiyuan ZHANG</strong></a>
    ·
    <a href="."><strong>Tang RUI</strong></a>
    ·
    <a href="https://hongbofu.people.ust.hk/"><strong>Hongbo FU<strong></a>
    ·
    <a href="https://scholars.cityu.edu.hk/en/persons/jingliao/"><strong>Jing LIAO</strong></a>
  </p>
  <h2 align="center">SIGGRAPH 2026 (Conference)</h2>
  <div align="center">
    <img src="assets/teaser.jpg", width="80%">
  </div>
</p>

This repository contains the code for the paper Controllable Texture Tiling with Transformed RoPE-Enhanced Diffusion Models , accepted as a conference paper in SIGGRAPH 2026.

## Setup
The code is written in python 3.10 and uses PyTorch along with pytorch-lightning for distributed training.
Please create a new conda environment with python 3.10 and run the `setup.sh` script to install all the dependencies.

```bash
conda create -n ctile python=3.10
conda activate ctile

sh setup.sh
```

## Dataset & Pretrained Models

Our dataset can be downloaded [here](https://modelscope.cn/datasets/heika94/ctile-dataset) and be stored in `dataset` folder.

We also uploaded the baseline dataset [here](https://modelscope.cn/datasets/heika94/ctile-baseline), you can also store it in `dataset` folder.

The pretrained modelscanbe downloaded [here](.) and stored in `ckpt` folder.

After running the setup script and placing the dataset and checkpoints in the correct location, the directory structure should look like this:

```
    .
    |-- ControlTile <-- this repository
    |   |-- dataset
    |   |   |-- ctile-dataset
    |   |   |   |-- metadata
    |   |   |   |-- render
    |   |   |   |-- texture
    |   |   |   |-- texture
    |   |   |
    |   |   |-- ctile-baseline
    |   |       |-- metadata
    |   |       |-- raw
    |   |       |-- synthesis
    |   |       |-- ratexture
    |
    |   |-- ckpt
    |       |-- model-lora.ckpt
    |   |-- output
    |       |-- logs
    |       |-- ckpt
    |   |-- ALL OTHER CONTENTS OF THIS REPOSITORY
```

## Training

```bash
# train with blender dataset
python run.py train_phase1
# train with adaptation dataset, remember to resume from phase1
python run.py train_phase2
```

## Testing

```bash
python run.py test
```

## Acknowledgements

We would like to express our sincere gratitude to **Manycore Tech Inc.** for providing the high-quality, photo-realistic datasets via [Spatial-Verse](https://spatial-verse.com/) that supported this research and development.

## Citation

```bibtex
TBD
```


