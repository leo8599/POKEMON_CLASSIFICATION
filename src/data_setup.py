"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
  """Creates training and testing DataLoaders."""
  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes
  test_class_names = test_data.classes

  # CRITICAL SAFETY CHECK
  # This ensures the indices match. If Train has 151 classes and Test has 150, this stops execution.
  if len(class_names) != len(test_class_names):
      print(f"[ERROR] Mismatch in class counts!")
      print(f"Train classes: {len(class_names)}")
      print(f"Test classes: {len(test_class_names)}")
      raise ValueError("Train and Test directories must have the exact same sub-directories (classes).")

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names