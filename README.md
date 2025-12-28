# Chinese-English NMT

This repository provides an implementation of Neural Machine Translation (NMT) models using both RNN-based and Transformer-based architectures. The models can be trained using custom datasets, and the training configurations are controlled via the `config/config.yaml` file. The repository also supports inference and evaluation of trained models on test datasets.

## Requirements

Before running the code, ensure you have the following installed:

```
pip install torch transformers tqdm pyyaml sacrebleu sentencepiece safetensors
```

## Dataset

The `data` directory contains the following files:

- `train_100k.jsonl` : Training dataset.
- `valid.jsonl` : Validation dataset.
- `test.jsonl` : Test dataset.

Each entry in these datasets should be a JSON object with the following fields:

- `zh`: Chinese sentence.
- `en`: English translation.

## Training

You can train both the RNN-based and Transformer-based models using the `train.py` script. The training configuration is controlled via the `config/config.yaml` file.

To start training the models:

```
python train.py
```

The `config/config.yaml` file includes parameters such as:

- Model architecture (RNN or Transformer)
- Learning rate
- Batch size
- Other model-specific hyperparameters

## Inference

You can use the `inference.py` script for inference and evaluation of the trained model on the test dataset.

To run inference:

```
python inference.py
```

This script will generate translations for the sentences in the test set and compute various evaluation metrics, such as:

- **BLEU** score
- **Perplexity** (Fluency)

## T5 Fine-Tuning

The `t5_finetune.py` script allows you to fine-tune a T5 model for translation tasks. The fine-tuned model can be used to perform translation tasks with the same pipeline as the RNN and Transformer models.

To fine-tune T5:

```
python t5_finetune.py
```

To run inference:

```
python t5_inference.py
```

## Pre-trained Models

Trained models can be found in the following Google Drive folder:

[Google Drive: Pre-trained Models](https://drive.google.com/drive/folders/1iv5vX7xV4IlhCUSevdGitGhRw8mHKybK?usp=drive_link)

The models are saved as checkpoints and can be directly loaded into the code for inference or further fine-tuning.
