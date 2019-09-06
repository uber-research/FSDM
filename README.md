Flexibly-Structured Model for Task-Oriented Dialogues
=====================================================

This repository contains the code of the SIGDIAL 2019 paper:

Lei Shu, Piero Molino, Mahdi Namazifar, Hu Xu, Bing Liu, Huaixiu Zheng, Gokhan Tur

[Flexibly-Structured Model for Task-Oriented Dialogues](https://arxiv.org/abs/1908.02402)

Here is the [slides](https://leishu02.github.io/FSDM_SIGDIAL2019.pdf).

## Problem to Solve
We proposes a novel end-to-end architecture for task-oriented dialogue systems.
It is based on a simple and practical yet very effective sequence-to-sequence approach, where language understanding and state tracking tasks are modeled jointly with a structured copy-augmented sequential decoder and a multi-label decoder for each slot.
The policy engine and language generation tasks are modeled jointly following that.
The copy-augmented sequential decoder deals with new or unknown values in the conversation, while the multi-label decoder combined with the sequential decoder ensures the explicit assignment of values to slots.
On the generation part, slot binary classifiers are used to improve performance.
This architecture is scalable to real-world scenarios and is shown through an empirical evaluation to achieve state-of-the-art performance on both the Cambridge Restaurant dataset and the Stanford in-car assistant dataset.


## Instructions

Please download glove embedding `glove.6B.50d.txt` and place it under `data/glove/`

### Model training
for camrest dataset: `python model.py -mode train -data camrest`
for kvret dataset: `python model.py -mode train -data kvret`

### Model testing
for camrest dataset: `python model.py -mode test -data camrest`
for kvret dataset: `python model.py -mode test -data kvret`

### Model finetuning
for camrest dataset: `python model.py -mode adjust -data camrest`
for kvret dataset: `python model.py -mode adjust -data kvret`

### Hyperparameter configuration

In order to configure hypermeters change the values in `config.py` or use the `-cfg` argument:
`python model.py -mode adjust -data camrest -cfg epoch_num=50 beam_search=True`

### Dataset
We evaluate the FSDM on two datasets: Cambridge Restaurant dataset (CamRest) and the Stanford in-car assistant dataset (KVRET).
We place them under `data/` folder. If you use the data, please also cite their papers.

## Citing

If you use the code, please cite:

```
@misc{shu2019flexiblystructured,
    title={Flexibly-Structured Model for Task-Oriented Dialogues},
    author={Lei Shu and Piero Molino and Mahdi Namazifar and Hu Xu and Bing Liu and Huaixiu Zheng and Gokhan Tur},
    year={2019},
    eprint={1908.02402},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
