Flexibly-Structured Model for Task-Oriented Dialogues
=====================================================

This repository contains the code of the SIGDIAL 2019 paper:

Lei Shu, Piero Molino, Mahdi Namazifar, Hu Xu, Bing Liu, Huaixiu Zheng, Gokhan Tur

[Flexibly-Structured Model for Task-Oriented Dialogues](https://arxiv.org/abs/1908.02402)


Instructions
------------

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

### Citing

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
