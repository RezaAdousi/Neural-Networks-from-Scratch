# 02 - Character-Level Language Modeling (Makemore)

This folder contains implementations and experiments for character-level language modeling, inspired by Andrej Karpathy's `makemore` series.

### Key Concepts Implemented:

* **N-gram Language Models:** Building bigram and trigram models to predict the next character from previous character context.

* **Neural Network Language Modeling:** Implementing character-level language models using trainable weight matrices, logits, probability distributions, and gradient descent.

* **Training & Evaluation:** Splitting the dataset into train, development, and test sets and evaluating models using negative log-likelihood loss.

* **Regularization:** Experimenting with different regularization strengths and using the development set to select the best setting.

* **PyTorch Fundamentals:** Working with one-hot encoding, tensor indexing, matrix multiplication, automatic differentiation, and `F.cross_entropy`.

* **Efficient Implementation:** Replacing explicit one-hot encoding with direct indexing into the model's weight matrix.
