import os
import shutil

# ask for the name of the data folder
data_folder = input("Enter the name of the data folder: ")

# ask for the name of the new directory to save the training and test data folders
new_folder = input("Enter the name of the new directory to save the training and test data folders: ")

# create the new directory if it doesn't exist
if not os.path.exists(new_folder):
    os.makedirs(new_folder)

# loop through the class subfolders in the data folder
for class_folder in os.listdir(data_folder):
    class_path = os.path.join(data_folder, class_folder)

    # create training and test folders for the current class
    train_folder = os.path.join(new_folder, "train", class_folder)
    test_folder = os.path.join(new_folder, "test", class_folder)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # get the filenames in the class folder
    filenames = os.listdir(class_path)
    num_files = len(filenames)

    # split the filenames into training and test sets
    train_size = int(num_files * 0.9)
    train_filenames = filenames[:train_size]
    test_filenames = filenames[train_size:]

    # copy the training and test files to the respective folders
    for filename in train_filenames:
        src_path = os.path.join(class_path, filename)
        dst_path = os.path.join(train_folder, filename)
        shutil.copy(src_path, dst_path)

    for filename in test_filenames:
        src_path = os.path.join(class_path, filename)
        dst_path = os.path.join(test_folder, filename)
        shutil.copy(src_path, dst_path)

print("Training and test data folders created in", new_folder)
