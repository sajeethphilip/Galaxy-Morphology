import os
import pickle
import json
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ask user for path to data folder
data_folder = input("Enter path to data folder: ")

# get the subfolders representing the class labels
class_labels = sorted(os.listdir(data_folder))

# create empty list to store flattened arrays
flattened_arrays = []

# loop over subfolders to flatten pickle files and add to list
for label in class_labels:
    data_path = os.path.join(data_folder, label)
    pkl_files = os.listdir(data_path)
    label_arrays = [pickle.load(open(os.path.join(data_path, f), "rb")).flatten() for f in pkl_files]
    flattened_arrays.extend(label_arrays)

# load the saved model from JSON file
model_file = input("Enter the path to the model file (.json extension): ")
with open(model_file, "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)

# check if there is a saved weights file and if so, load it
weights_file = model_file.replace(".json", ".h5")
best_model_file = model_file.replace(".json", "_best.h5")
if os.path.exists(best_model_file):
    model.load_weights(best_model_file)
    print("Loaded saved model weights from", best_model_file,"for the model",model_file)

# split the data into training and testing sets
X = np.array(flattened_arrays)
if model.output_shape[-1] == 1:
    # if model output is binary, use labels with values 0 or 1
    y = np.array([class_labels.index(os.path.basename(label)) for _ in range(len(pkl_files)) for label in pkl_files])
else:
    # if model output is not binary, use one-hot encoded labels
    # create a dictionary to map each label to an integer
    label_to_int = {label: i for i, label in enumerate(class_labels)}

    # create empty list to store y values
    y = []

    # loop over subfolders to get list of pkl files
    for label in class_labels:
        data_path = os.path.join(data_folder, label)
        pkl_files = os.listdir(data_path)

        # loop over pkl files to get their labels and append to y list
        for f in pkl_files:
            # check if pkl file exists
            if os.path.isfile(os.path.join(data_path, f)):
                # get label for this pkl file
                label_int = label_to_int.get(label, -1)
                if label_int == -1:
                    # if label is not found in the dictionary, ignore this pkl file
                    continue
                if model.output_shape[-1] == 1:
                    # if model output is binary, use labels with values 0 or 1
                    y.append(label_int)
                else:
                    # if model output is not binary, use one-hot encoded labels
                    y.append(to_categorical(label_int, len(class_labels)))

    # convert y list to numpy array
    y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# compile the model with appropriate loss function
if model.output_shape[-1] == 1:
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
else:
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model
best_val_acc = 0.0
patience = 10
epochs = 500
for i in range(epochs):
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=32, verbose=1)
    val_acc = history.history['val_accuracy'][0]
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # save the best model based on validation accuracy
        if os.path.exists(best_model_file):
            os.remove(best_model_file)
        model.save_weights(best_model_file)
        print(f"Saved best model weights with validation accuracy {best_val_acc:.4f} to {best_model_file}")
    else:
        if os.path.exists(weights_file):
            os.remove(weights_file)
        model.save_weights(weights_file)
        print(f"Saved current weights with validation accuracy {val_acc:.4f} to {weights_file}")

    if history.history['accuracy'][0] == 1.0:
        print("Training accuracy reached 1.0, stopping training")
        break
