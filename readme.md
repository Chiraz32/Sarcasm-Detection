# Sarcasm Detection

This project demonstrates a sarcasm detection model using a Long Short-Term Memory (LSTM) neural network. The model processes textual input to classify it as sarcastic or non-sarcastic. The notebook includes data preprocessing, model training, evaluation, testing, and visualization steps.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)

## Project Overview

Sarcasm detection is crucial in natural language processing (NLP), especially for applications in social media sentiment analysis, chatbots, and opinion mining. Since sarcasm often conveys an opposite meaning from the literal words, it can mislead sentiment detection models. This project employs an LSTM model to learn sequential dependencies in text data for identifying sarcastic language.

## Dataset

The notebook uses labeled datasets containing sarcastic and non-sarcastic text samples from sources such as Reddit, including:

- **DWAEF-sarc**
- **GEN-sarc**
- **HYP-sarc**
- **RQ-sarc**
- **REDDIT-sarc**

### Dataset Preprocessing

The preprocessing steps include:

1. **Noise Removal**: Removing special characters, URLs, and non-alphanumeric tokens to reduce noise.
2. **Tokenization and Padding**: Tokenizing the text and padding sequences for uniform input length.
3. **Label Encoding**: Converting sarcasm labels into a binary format for classification.

## Requirements

The following libraries are required to run the notebook:

```bash
pip install numpy pandas torch scikit-learn matplotlib nltk seaborn
```

## Usage

1. **Load Dataset**: Import and load the dataset into a Pandas DataFrame.
2. **Preprocess Data**: Execute data cleaning, tokenization, padding, and label encoding.
3. **Build Model**: Define an LSTM model with an embedding layer, LSTM layer, and dense output layer for binary classification.
4. **Train Model**: Train the model on the training dataset with cross-validation.
5. **Evaluate Results**: Calculate performance metrics and visualize the results.

To run the notebook:

```bash
jupyter notebook lstm-sarcasm-detection.ipynb
```

## Model Architecture

The LSTM model architecture consists of:

- **Embedding Layer**: Converts tokens into dense vectors, capturing semantic meaning.
- **LSTM Layer**: Processes sequences, learning dependencies over time steps to detect contextual cues for sarcasm.
- **Dense Layer**: Outputs a binary classification for sarcastic and non-sarcastic labels.

## Training and Evaluation

The model undergoes **10-fold cross-validation** to ensure reliable performance metrics. For each fold:

1. The model is trained on a subset of data, and validation is performed on the remaining set.
2. **Metrics Calculated Per Fold**:
   - **Accuracy**: Measures correct predictions over total predictions.
   - **Precision**: Indicates the accuracy of sarcasm predictions.
   - **Recall**: Reflects the model's capability to identify all sarcastic samples.
   - **F1 Score**: Combines precision and recall for balanced evaluation.

### Average Results Across Folds

- **Accuracy**: 70.97%
- **Precision**: 72%
- **Recall**: 69%
- **F1 Score**: 70%

These metrics are computed and averaged to assess the model's ability to generalize across different subsets.
