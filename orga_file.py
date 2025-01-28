import os
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToPILImage

# Paths
base_dir = "data/cifar10_correct"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "test")
train_list_file = "datalists/cifar10_train.txt"
val_list_file = "datalists/cifar10_test.txt"

# Ensure base directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Download CIFAR-10 dataset
cifar10_dataset = CIFAR10(root="data", train=True, download=True)

# Define class names
class_names = cifar10_dataset.classes

# Create subdirectories for each class
for class_name in class_names:
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

# Transform to save images as PNG
to_pil = ToPILImage()

# Map class indices to class names
class_to_indices = {class_name: [] for class_name in class_names}
for idx, (_, label) in enumerate(cifar10_dataset):
    class_name = class_names[label]
    class_to_indices[class_name].append(idx)

# Save images to the structure specified in the datalist file
def save_images_from_list(datalist_file, target_dir, dataset, class_to_indices):
    with open(datalist_file, "r") as f:
        for line in f:
            # Parse the relative path from the file
            relative_path = line.strip()
            class_name, image_name = relative_path.split("/")[1], relative_path.split("/")[2]

            # Get the sequential index from the file name (e.g., '0001.png' -> 0-based index 0)
            sequential_index = int(image_name.split(".")[0]) - 1

            # Fetch the dataset index using the class_to_indices mapping
            dataset_index = class_to_indices[class_name][sequential_index]

            # Fetch the image and label from CIFAR-10
            image, label = dataset[dataset_index]

            # Save the image to the target directory
            class_dir = os.path.join(target_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            file_path = os.path.join(class_dir, image_name)
            image.save(file_path)

# Process training and validation lists
save_images_from_list(train_list_file, train_dir, cifar10_dataset, class_to_indices)

# If you have a separate validation dataset, load it and process similarly
val_dataset = CIFAR10(root="data", train=False, download=True)
class_to_indices_val = {class_name: [] for class_name in class_names}
for idx, (_, label) in enumerate(val_dataset):
    class_name = class_names[label]
    class_to_indices_val[class_name].append(idx)

save_images_from_list(val_list_file, val_dir, val_dataset, class_to_indices_val)

print(f"CIFAR-10 dataset organized based on '{train_list_file}' and '{val_list_file}'!")
