"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms

def main():
    # Setup hyperparameters
    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    HIDDEN_UNITS = 15
    LEARNING_RATE = 0.001

    # Setup directories
    train_dir = "data/train"
    test_dir = "data/test"
    
    # *** NEW ***
    # Setup model save path
    MODEL_NAME = "05_going_modular_script_mode_tinyvgg_model.pth"
    MODEL_SAVE_DIR = "models"
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create transforms
    data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )

    # Create model with help from model_builder.py
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    ).to(device)

    # *** NEW ***
    # Check if a saved model exists and load it
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"[INFO] Loading existing model from: {MODEL_SAVE_PATH}")
        try:
            # Load the saved state_dict
            # We use map_location to ensure the model loads correctly onto the 'device'
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
            print("[INFO] Model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            print("[INFO] Training from scratch.")
    else:
        print(f"[INFO] No existing model found at {MODEL_SAVE_PATH}. Training from scratch.")
    # *** END NEW ***

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters()
                                 #,lr=LEARNING_RATE
                                 )

    # Start training with help from engine.py
        
    engine.train(model=model,
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 loss_fn=loss_fn,
                 optimizer=optimizer,
                 epochs=NUM_EPOCHS,
                 device=device)

    # Save the model with help from utils.py
    # *** NEW (using variables) ***
    utils.save_model(model=model,
                     target_dir=MODEL_SAVE_DIR,
                     model_name=MODEL_NAME)

if __name__ == "__main__":
    main()