Predicting Dyslexia based on eye-movements

Create a conda environment with
```bash
$ conda env create -f environment.yml
```
Then activate the environment and install your appropriate version of [PyTorch](https://pytorch.org/get-started/locally/).
```bash
$ conda install -y pytorch torchvision cudatoolkit=11.1 -c pytorch
$ # conda install pytorch torchvision cpuonly -c pytorch
$ pip install datasets transformers
```
