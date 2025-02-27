# PIONEER (Platform for Iterative Optimization and Navigation to Enhance Exploration of Regulatory sequences)

This repository contains a PyTorch implementation for the paper "PIONEER:  An in silico playground for iterative improvement of genomic deep learning". 

## Design Choices

This code is meant to be FAIR (findable, accessible, interoperable, and reusable).
The primary files are:

1. ```pioneer.py```: the main PIONEER class.
2. ```scoring.py```: the Scorer class, to attribute a score to given sequences.
3. ```generator.py```: the Generator class, to generate sequences from an original source pool of sequences.
4. ```selector.py```: the Selector class, to perform a selection from a pool of sequences.
5. ```oracle.py```: the Oracle class, to assign labels to sequences.
6. ```surrogate.py```: the Surrogate class, i.e. the model that has to be trained at every cycle.

## Installation and setup

```
conda env create -f environment.yml
```

which will create a ```pioneer``` environment with packages installed (please provide your server username in place of ```<username>```). 

To import library:

```
import pioneer
```


<!-- ### Datasets and Oracles

We provide preprocessed datasets for [DeepSTARR](https://huggingface.co/datasets/anonymous-3E42/DeepSTARR_preprocessed), and [LentiMPRA](). -->

