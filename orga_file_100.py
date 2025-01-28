import os
import shutil
from torchvision.datasets import CIFAR100
from torchvision import transforms
from PIL import Image

# Define the root directory for the dataset
root_dir = "./data/cifar100"
train_dir = os.path.join(root_dir, "train")
test_dir = os.path.join(root_dir, "test")

# Create the directory structure
def create_directory_structure():
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)  # Clear existing structure if any
    os.makedirs(train_dir)
    os.makedirs(test_dir)

    # Get the class names from CIFAR-100
    class_names = CIFAR100(root="./", download=True).classes

    # Create subdirectories for each class in train and test
    for class_name in class_names:
        os.makedirs(os.path.join(train_dir, class_name))
        os.makedirs(os.path.join(test_dir, class_name))

# Save images to the corresponding class folder
def save_images(dataset, target_dir, prefix):
    image_paths = []
    for idx, (image, label) in enumerate(dataset):
        class_name = dataset.classes[label]
        class_dir = os.path.join(target_dir, class_name)

        # Save the image to the appropriate directory
        image_path = os.path.join(class_dir, f"img_{idx}.png")
        image.save(image_path)
        relative_path = f"{prefix}/{class_name}/img_{idx}.png {label}"
        image_paths.append(relative_path)

    return image_paths

# Create text files with image paths
def create_path_files(train_paths, test_paths):
    train_txt_path = os.path.join(root_dir, "cifar100_train.txt")
    test_txt_path = os.path.join(root_dir, "cifar100_test.txt")

    with open(train_txt_path, "w") as train_file:
        for path in train_paths:
            train_file.write(f"{path}\n")

    with open(test_txt_path, "w") as test_file:
        for path in test_paths:
            test_file.write(f"{path}\n")

# Main function
def main():
    # Create the directory structure
    create_directory_structure()

    # Define the transformations for the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),  # Convert back to PIL for saving
    ])

    # Load the training and test datasets
    train_dataset = CIFAR100(root="./", train=True, download=True, transform=transform)
    test_dataset = CIFAR100(root="./", train=False, download=True, transform=transform)

    # Save images to the appropriate folders
    print("Saving training images...")
    train_paths = save_images(train_dataset, train_dir, "train")

    print("Saving test images...")
    test_paths = save_images(test_dataset, test_dir, "test")

    # Create the text files with image paths
    print("Creating path files...")
    create_path_files(train_paths, test_paths)

    print("Dataset organization complete.")

if __name__ == "__main__":
    main()
