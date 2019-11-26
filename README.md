Flexibly-Structured Model for Task-Oriented Dialogues
=====================================================

This repository contains the code of the SIGDIAL 2019 paper:

Lei Shu, Piero Molino, Mahdi Namazifar, Hu Xu, Bing Liu, Huaixiu Zheng, Gokhan Tur

[Flexibly-Structured Model for Task-Oriented Dialogues](https://arxiv.org/abs/1908.02402)

Here are the [slides](https://leishu02.github.io/FSDM_SIGDIAL2019.pdf).

## FSDM

FSDM a novel end-to-end architecture for task-oriented dialogue systems.
It is based on a simple and practical yet very effective sequence-to-sequence approach, where language understanding and state tracking tasks are modeled jointly with a structured copy-augmented sequential decoder and a multi-label decoder for each slot.
The policy engine and language generation tasks are modeled jointly following that.
The copy-augmented sequential decoder deals with new or unknown values in the conversation, while the multi-label decoder combined with the sequential decoder ensures the explicit assignment of values to slots.
On the generation part, slot binary classifiers that predict if a slot will appear in the answer are used to improve performance.
This architecture is scalable to real-world scenarios and is shown through an empirical evaluation to achieve state-of-the-art performance on both the Cambridge Restaurant dataset and the Stanford in-car assistant dataset.


## Instructions

Please download GloVe embedding `glove.6B.50d.txt` from [GloVe website](https://nlp.stanford.edu/projects/glove/) and place them under `data/glove/`.

### Dataset
The [CamRest676](https://www.repository.cam.ac.uk/handle/1810/260970) and [Stanford KVRET in-car assistant](https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/) datasets are provided in a preprocessed JSON format for convenience, but they belong to the original authors.
Please download and place them under `data/CamRest676` and `data/kvret` respectively.

 
### Model training
For camrest dataset: `python model.py -mode train -data camrest`
For kvret dataset: `python model.py -mode train -data kvret`

### Model testing
For camrest dataset: `python model.py -mode test -data camrest`
For kvret dataset: `python model.py -mode test -data kvret`

### Model finetuning
For camrest dataset: `python model.py -mode adjust -data camrest`
For kvret dataset: `python model.py -mode adjust -data kvret`

### Hyperparameter configuration

In order to configure hypermeters change the values in `config.py` or use the `-cfg` argument:
`python model.py -mode adjust -data camrest -cfg epoch_num=50 beam_search=True`

## Citing

If you use the code, please cite:

```
@inproceedings{shu-etal-2019-flexibly,
    title = "Flexibly-Structured Model for Task-Oriented Dialogues",
    author = "Shu, Lei  and
      Molino, Piero  and
      Namazifar, Mahdi  and
      Xu, Hu  and
      Liu, Bing  and
      Zheng, Huaixiu  and
      Tur, Gokhan",
    booktitle = "Proceedings of the 20th Annual SIGdial Meeting on Discourse and Dialogue",
    month = sep,
    year = "2019",
    address = "Stockholm, Sweden",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W19-5922",
    pages = "178--187"
}
```
