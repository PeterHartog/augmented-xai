# Using test-time augmentation to investigate explainable AI: inconsistencies between method, model and human intuition

- [Introduction](#Introduction)
- [Installation](#installation)
- [Examples](#examples)
- [References](#references)
- [License](#license)

## Introduction

Stakeholders of machine learning models desire explainable artificial intelligence (XAI) to produce human-understandable and consistent interpretations. In computational toxicity, augmentation of text-based molecular representations has been used successfully for transfer learning on downstream tasks. Augmentations of molecular representations can also be used at inference to compare differences between multiple representations of the same ground-truth.
In this study, we investigate the robustness of eight XAI methods using test-time augmentation for a molecular-representation model in the field of computational toxicity prediction.
We report significant differences between explanations for different representations of the same ground-truth, and show that randomized models have similar variance. We hypothesize that text-based molecular representations in this and past research reflect tokenization more than learned parameters. Furthermore, we see a greater variance between in-domain predictions than out-of-domain predictions, indicating XAI measures something other than learned parameters. Finally, we investigate the relative importance given to expert-derived structural alerts and find similar importance given irregardless of applicability domain, randomization and varying training procedures. We therefore caution future research to validate their methods using a similar comparison to human intuition without further investigation.

## Installation

### Installation (python v 3.11.5)

Install [pytorch](https://pytorch.org/get-started/locally/) 2.1.1 according to your system

```bash
pip install torch torchvision torchaudio
```

Install [registry-factory](https://github.com/aidd-msca/registry-factory) and [aidd-codebase](https://github.com/aidd-msca/aidd-codebase) and other requirements

```bash
pip install -r requirements.txt
```

## Examples

### Data Processing

Quick start using the same parameters of the original publication.

```bash
python smiles_cleaning.py
```

To download and clean the Ames dataset from Therapeutics Data Commens with other parameters:

```bash
python representation/prepare_data.py experiment=data_cleaning/ames
```

This creates a ames_cleaned.csv file. In the notebook ames_data_analysis, the various files are created, run that notebook if you wish to remake the original data files.

### Training the models

Then you can train a model with a specific experiment with the following command:

```bash
python representation/train.py experiment=xxx
```

In the experiment you can train various models, including pre-training on chembl with experiment=pretrain/enc_dec/ME2C or transfer-learn with experiment=ames_training/NN/enc_dec/ME2C

### Datasets

- ChEMBL [[1]](#1)[[2]](#2)
- Ames [[3]](#3)

### Experiments

#### pre-training

- BERT-style
  - C2C
  - R2C
  - E2C
  - MC2C
  - MR2C
  - ME2C
- BART-style
  - C2C
  - R2C
  - E2C
  - MC2C
  - MR2C
  - ME2C

#### Transfer Learning

- Transformer CNN
  - enc_dec
  - enc_only
- Transformer NN
  - enc_dec
  - enc_only

## Cite this repository

Awaiting Review...

## References

<a id="1">[1]</a> : Davies, M., Nowotka, M., Papadatos, G., Dedman, N., Gaulton, A., Atkinson,
F., Bellis, L., Overington, J.P.: Chembl web services: streamlining access to drug
discovery data and utilities. Nucleic acids research 43(W1), 612–620 (2015)

<a id="2">[2]</a> : Mendez, D., Gaulton, A., Bento, A.P., Chambers, J., De Veij, M., F ́elix, E., Mag-
ari ̃nos, M.P., Mosquera, J.F., Mutowo, P., Nowotka, M., et al.: Chembl: towards
direct deposition of bioassay data. Nucleic acids research 47(D1), 930–940 (2019)

<a id="3">[3]</a> : Xu, C., Cheng, F., Chen, L., Du, Z., Li, W., Liu, G., Lee, P.W., Tang, Y.: In
silico prediction of chemical ames mutagenicity. Journal of chemical information
and modeling 52(11), 2840–2847 (2012)
