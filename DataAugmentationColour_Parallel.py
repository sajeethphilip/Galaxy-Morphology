import os
from PIL import Image, ImageEnhance, ImageOps, ImageStat
import numpy as np
import concurrent.futures


# Define the augmentation functions (same as before)
# Define the augmentation functions
def augment_brightness(image):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(np.random.randint(1, 5))

def augment_contrast(image):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(np.random.randint(1, 5))

def augment_sharpness(image):
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(np.random.randint(-1, 5))

def augment_flip(image):
    return image.transpose(method=Image.FLIP_LEFT_RIGHT)

def augment_invert(image):
    return ImageOps.invert(image)

def augment_rotate(image):
    angle = np.random.randint(-90, 90)
    median_color = ImageStat.Stat(image).median
    return image.rotate(angle, fillcolor=tuple(map(int, median_color)))

# Define the image file extensions to be processed (same as before)

# Define the image file extensions to be processed
extensions = ('.jpg', '.jpeg', '.png','.gif','.bmp')

# Define the parent directory to read images from
parent_dir = input("Specify the path to the images to be augmented:")

# Function to apply augmentations and save augmented images
def process_image(file_path):
    with Image.open(file_path) as image:
        basename, extension = os.path.splitext(file_path)
        augmented_images = []

        for augmentation_name, augmentation_function in [('R', augment_rotate),('b', augment_brightness), ('c', augment_contrast), ('s', augment_sharpness), ('f', augment_flip), ('i', augment_invert)]:
            augmented_image = augmentation_function(image)
            output_file_name = f"{basename}_{augmentation_name}{extension.replace('.jpeg', '').replace('.jpg', '')}.png"
            output_file_path = os.path.join('', output_file_name)
            augmented_image.save(output_file_path)
            augmented_images.append(output_file_path)

        # Apply translational and scaling invariance (same as before)

        return augmented_images

# Function to process images in parallel
def process_images_parallel(file_paths):
    augmented_images = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_image, file_path) for file_path in file_paths]

        for future in concurrent.futures.as_completed(futures):
            try:
                augmented_images.extend(future.result())
            except Exception as e:
                print(f"An error occurred: {e}")

    return augmented_images

# Loop through all files and subfolders in the parent directory
file_paths = []
for root, dirs, files in os.walk(parent_dir):
    for file_name in files:
        if file_name.lower().endswith(extensions):
            file_path = os.path.join(root, file_name)
            file_paths.append(file_path)

# Process images in parallel
augmented_images = process_images_parallel(file_paths)

# You can use the list of augmented image paths for further processing if needed
print("Augmentation completed successfully.")
