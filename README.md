# BERT-Based Text Classification Project

This project involves fine-tuning a BERT model to classify sequences of text into predefined categories. The project is structured with three main scripts: `train.py`, `inference.py`, and `main.py`. Each script has a specific role in the overall process, from training the model to making predictions and interacting with an API.

## Table of Contents

- [Project Overview](#project-overview)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Performing Inference](#performing-inference)
  - [Interacting with the API](#interacting-with-the-api)

## Project Overview

The purpose of this project is to create a text classification model using BERT. The model is trained on a dataset of text sequences and their corresponding labels. After fine-tuning, the model can predict the label of new, unseen text sequences.

### Key Components:
- **Training the Model:** Fine-tune a pre-trained BERT model on your dataset.
- **Inference:** Use the trained model to classify new text sequences.
- **API Interaction:** Send text sequences to an API and process the responses.

## Setup and Installation

### Prerequisites

- Python 3.6+
- PyTorch
- Hugging Face Transformers
- Pandas
- scikit-learn
- requests


## Usage

### 1. Training the Model

To train the model, run the `train.py` script. This will fine-tune a BERT model on your dataset.

```bash
python train.py
