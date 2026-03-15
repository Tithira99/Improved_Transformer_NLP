# Transformer-Based Neural Machine Translation Improvements
## Project Information
- Development Period: November 2025 
- Course: Natural Language Processing with Deep Learning
- Type: Team project done at the University of Aizu

## Project Overview
This project explores improvements to a Transformer-based Neural Machine Translation (NMT) model for Japanese → English translation and applies the improved model to Sinhala → English translation.

The base implementation of the Transformer model was provided as part of the course framework. The project focused on analyzing the baseline model, identifying weaknesses, and improving translation performance through several modifications and optimizations.

## Key Improvements

Several issues were identified in the baseline Transformer implementation and addressed through improvements:

- Corrected the tokenization pipeline using appropriate tools for each language (MeCab for Japanese and NLTK for English).

- Fixed incorrect source–target data ordering in the training pipeline.

- Added model checkpointing and loading mechanisms to avoid unnecessary retraining.

- Increased the training dataset size from 25,000 to 100,000 samples.

- Increased the number of training epochs to improve model convergence.

- Implemented Beam Search decoding with length penalty to improve translation quality.

These improvements resulted in **better translation outputs and higher BLEU scores compared to the baseline model.**

## Extension to a New Language

The improved Transformer architecture was also applied to **Sinhala → English** translation using the **OPUS100 dataset**.

To further improve efficiency, **Dynamic Sparse Training (DST)** was explored as a method to reduce computational cost while maintaining translation performance.

Experimental results showed that DST-based models could achieve **faster computation while maintaining competitive translation accuracy.**

## My Contributions

- Coordinated the overall project work

- Analyzed the baseline Transformer implementation

- Identified weaknesses in the model pipeline

- Implemented improvements to the JP → EN translation model

- Performed debugging, testing, and experimentation

- Created visualizations for model performance and attention analysis

- Prepared presentation materials and presented the results

# Technologies Used

- Python

- PyTorch

- Transformer Architecture

- Neural Machine Translation (NMT)

- MeCab (Japanese Tokenization)

- NLTK

- SentencePiece

- Beam Search Decoding

# Future Improvements

Further improvements to translation quality may include:

- Using larger and cleaner datasets

- Applying advanced preprocessing and data filtering

- Training on larger OPUS100 datasets

- Increasing training epochs and model tuning
