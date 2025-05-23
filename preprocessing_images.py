# Malota SP 41370015

# Import necessary libraries
import os
from PIL import Image, UnidentifiedImageError  # For image loading and error handling
from torchvision import transforms  # For image preprocessing transformations
from torch.utils.data import Dataset, DataLoader  # For dataset and dataloader abstractions

# Define a custom dataset class for Intel image classification
class IntelImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir  # Root directory containing class folders
        self.transform = transform  # Transformations to apply to each image
        self.image_paths = []  # List of all image file paths
        self.labels = []  # List of corresponding labels
        self.class_names = []  # List of class names (folder names)
        self.class_to_idx = {}  # Mapping from class name to label index

        try:
            # Check if the directory exists
            if not os.path.exists(root_dir):
                raise FileNotFoundError(f"Directory not found: {root_dir}")

            # Get sorted list of class names (subdirectories)
            self.class_names = sorted([
                name for name in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, name))
            ])
            
            # Raise error if no class folders found
            if not self.class_names:
                raise ValueError(f"No class folders found in {root_dir}")
            
            # Create mapping from class name to index
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}

            # Traverse class folders and collect image file paths and labels
            for cls_name in self.class_names:
                cls_folder = os.path.join(root_dir, cls_name)
                for fname in os.listdir(cls_folder):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):  # Valid image extensions
                        full_path = os.path.join(cls_folder, fname)
                        self.image_paths.append(full_path)
                        self.labels.append(self.class_to_idx[cls_name])

            # Raise error if no images found
            if not self.image_paths:
                raise ValueError(f"No valid image files found in {root_dir}")

        except Exception as e:
            print(f"Error initializing dataset from {root_dir}: {e}")
            raise

    def __len__(self):
        # Return the number of images in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and return a single image and its label
        try:
            image_path = self.image_paths[idx]
            label = self.labels[idx]
            image = Image.open(image_path).convert("RGB")  # Open and convert image to RGB
            if self.transform:
                image = self.transform(image)  # Apply transformations
            return image, label
        except UnidentifiedImageError:
            # Handle corrupted image
            print(f"Warning: Failed to load image {image_path}. It may be corrupted.")
            return self.__getitem__((idx + 1) % len(self))  # Try next image
        except Exception as e:
            print(f"Unexpected error loading image at index {idx}: {e}")
            raise

# Define image transformations for training and testing
def get_transforms(train=True, img_size=(150, 150)):
    if train:
        # Apply augmentations during training for better generalization
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        # Only resize and normalize during testing
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

# Main function to load datasets and preview one batch
def main():
    try:
        # Define paths to training and testing directories
        train_dir = "C:/Users/malot/OneDrive/Documents/AI/intel-image-ml-project/data/seg_train"
        test_dir = "C:/Users/malot/OneDrive/Documents/AI/intel-image-ml-project/data/seg_test"

        print("Loading datasets...")
        # Initialize training and testing datasets with transformations
        train_dataset = IntelImageDataset(train_dir, transform=get_transforms(train=True))
        test_dataset = IntelImageDataset(test_dir, transform=get_transforms(train=False))

        # Print basic dataset info
        print(f"Training samples: {len(train_dataset)}")
        print(f"Testing samples: {len(test_dataset)}")
        print(f"Classes found: {train_dataset.class_names}")

        # Create data loaders for batch processing
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Display one batch of training data
        for images, labels in train_loader:
            print("Batch image tensor shape:", images.shape)
            print("Batch labels:", labels)
            break  # Only show the first batch

    except Exception as e:
        print(f" An error occurred during preprocessing: {e}")

# Run main function when script is executed directly
if __name__ == "__main__":
    main()
