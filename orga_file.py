import os
import shutil
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToPILImage

# Paths
base_dir = "data/cifar10"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
train_list_file = "datalists/cifar10_train.txt"
val_list_file = "datalists/cifar10_test.txt"

# Ensure base directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Download CIFAR-10 dataset
cifar10_dataset = CIFAR10(root="data", download=True)

# Define class names
class_names = cifar10_dataset.classes

# Create subdirectories for each class
for class_name in class_names:
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

# Transform to save images as PNG
to_pil = ToPILImage()

# Save images to the structure specified in the datalist file
def save_images_from_list(datalist_file, target_dir):
    with open(datalist_file, "r") as f:
        for line in f:
            # Parse the relative path from the file
            relative_path = line.strip()
            class_name, image_name = relative_path.split("/")[1], relative_path.split("/")[2]
            
            # Get the image index from the name (e.g., '0001.png' -> 1)
            image_index = int(image_name.split(".")[0])
            
            # Fetch the image and label from CIFAR-10
            image, label = cifar10_dataset[image_index]
            
            # Save the image to the target directory
            class_dir = os.path.join(target_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            file_path = os.path.join(class_dir, image_name)
            image.save(file_path)

# Process training and validation lists
save_images_from_list(train_list_file, train_dir)
save_images_from_list(val_list_file, val_dir)

print(f"CIFAR-10 dataset organized based on '{train_list_file}' and '{val_list_file}'!")
