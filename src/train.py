"""
Trains a PyTorch image classification model using Transfer Learning.
"""

import os
import torch
import data_setup, engine, model_builder, utils
from torchvision import transforms
from timeit import default_timer as timer 

def main():
    # Setup hyperparameters
    NUM_EPOCHS = 10  # 10 epochs is usually enough for Transfer Learning
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    # Setup directories
    train_dir = "data/train"
    test_dir = "data/test"

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Mac M1/M2 support
    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        device = "mps"
    
    print(f"[INFO] Training on device: {device}")

    # Create transforms
    # IMPORTANT: EfficientNet requires 224x224 and specific normalization
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                             std=[0.5, 0.5, 0.5])
    ])

    # Create DataLoaders
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )

    # Create model (Transfer Learning)
    model = model_builder.create_efficientnet_b0(
        output_shape=len(class_names),
        device=device
    )

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Start training
    start_time = timer()
    
    results = engine.train(model=model,
                           train_dataloader=train_dataloader,
                           test_dataloader=test_dataloader,
                           loss_fn=loss_fn,
                           optimizer=optimizer,
                           epochs=NUM_EPOCHS,
                           device=device)
    
    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

    # Save the model
    utils.save_model(model=model,
                     target_dir="models",
                     model_name="pokemon_efficientnet_model.pth")
                     
    # Plot and save the results for your report
    utils.plot_loss_curves(results, output_path="pokemon_training_results.png")

if __name__ == "__main__":
    main()                                                    