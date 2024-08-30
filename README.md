Eye-tracking based classification of Mandarin Chinese readers with and without dyslexia using neural sequence models
====================================================================================================================
[![paper](https://img.shields.io/static/v1?label=paper&message=download%20link&color=brightgreen)](https://aclanthology.org/2022.tsar-1.10/)

This repository contains the sequence and baseline models used in Eye-tracking based classification of Mandarin Chinese readers with and without dyslexia using neural sequence models.

## Using the models

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

## Citation

Patrick Haller, Andreas Säuberli, Sarah Kiener, Jinger Pan, Ming Yan, and Lena Jäger. 2022. Eye-tracking based classification of Mandarin Chinese readers with and without dyslexia using neural sequence models. In Proceedings of the Workshop on Text Simplification, Accessibility, and Readability (TSAR-2022), pages 111–118, Abu Dhabi, United Arab Emirates (Virtual). Association for Computational Linguistics.
