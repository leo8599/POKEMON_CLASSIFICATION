# Pokemon Classification Project

A PyTorch-based image classification project that classifies Pokemon images using **Transfer Learning** with an **EfficientNet-B0** architecture.

> **Status:** Project Completed. Achieved **93.42% accuracy** on the test set using Fine-Tuning strategies.

## Dataset

The dataset is sourced from [Kaggle - Pokemon Classification](https://www.kaggle.com/datasets/lantian773030/pokemonclassification) and contains images of **150+ different Pokemon species** organized into training and testing sets.

## Project Structure

This project follows the modular PyTorch structure outlined in [PyTorch Going Modular](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/05_pytorch_going_modular.md).

```text
Pokemon_Classification/
├── data/
│   ├── train/               # Training images organized by Pokemon name
│   └── test/                # Testing images (generated via get_data.py)
├── models/                  # Saved model files (.pth)
├── src/
│   ├── data_setup.py        # DataLoader creation functionality
│   ├── engine.py            # Training and testing loops
│   ├── model_builder.py     # EfficientNet-B0 architecture setup
│   ├── train.py             # Phase 1: Feature Extraction Training
│   ├── train_cc.py          # Phase 2: Fine-Tuning Training (The accuracy booster)
│   ├── visualize_model.py   # Script to generate architecture diagrams
│   ├── utils.py             # Helper functions (saving models, plotting curves)
│   └── get_data.py          # Data downloading and split utilities
├── requirements.txt         # Project dependencies
└── README.md
````

## Model Architecture & Methodology

Instead of training a model from scratch (like TinyVGG), this project utilizes **Transfer Learning** to leverage the power of pre-trained networks.

  * **Base Architecture:** [EfficientNet-B0](https://arxiv.org/abs/1905.11946) (Pre-trained on ImageNet).
  * **Modifications:** The classification head was replaced with a custom sequence: `Dropout(p=0.2)` -\> `Linear(output_shape)`.
  * **Input Size:** 224x224 pixels (Normalized).

### Training Strategy (2 Stages)

1.  **Feature Extraction (`train.py`):**
      * **Frozen Layers:** All EfficientNet base layers were frozen.
      * **Training:** Only the classification head was trained.
      * **Result:** \~82% Accuracy.
2.  **Fine-Tuning (`train_cc.py`):**
      * **Unfrozen Layers:** All layers were unfrozen to adapt internal filters to Pokemon artistic styles.
      * **Hyperparameters:** Low learning rate (`1e-4`) to prevent catastrophic forgetting.
      * **Result:** **\~93.4% Accuracy**.

## Installation

1.  Clone the repository:

    ```bash
    git clone [https://github.com/leo8599/POKEMON_CLASSIFICATION.git](https://github.com/leo8599/POKEMON_CLASSIFICATION.git)
    cd POKEMON_CLASSIFICATION
    ```

2.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To reproduce the results, run the training in the following order:

### 1\. Feature Extraction (Base Training)

Trains the classifier head for 10 epochs.

```bash
cd src
python train.py
```

  * *Output:* Saves `models/pokemon_efficientnet_model.pth`

### 2\. Fine-Tuning (Performance Boost)

Loads the previous model, unfreezes layers, and refines weights for 5 epochs with a lower learning rate.

```bash
python train_cc.py
```

  * *Output:* Saves `models/pokemon_efficientnet_finetuned.pth`

### 3\. Visualization (Optional)

Generates a summary and diagram of the neural network architecture.

```bash
python visualize_model.py
```

## Results

| Metric | Feature Extraction | Fine-Tuning (Final) |
| :--- | :---: | :---: |
| **Test Accuracy** | \~82.00% | **93.42%** |
| **Test Loss** | \~0.78 | **0.27** |
| **Train Accuracy** | \~93.00% | **99.41%** |

*(Loss and Accuracy curves showing the improvement during the fine-tuning phase)*

## Credits & References

  * Modular structure inspired by **Daniel Bourke's** [PyTorch Going Modular](https://github.com/mrdbourke/pytorch-deep-learning).
  * EfficientNet implementation via `torchvision.models`.
  * Dataset from Kaggle.

<!-- end list -->

```
```
