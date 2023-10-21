import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os,cv2
import tensorflow as tf
# Step 6 Test the perormance
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
import os,pickle
from PIL import Image
import shutil
batch_size=128
mulfact=5
# Define the image size
img_size = (196,196)
if img_size==(196,196):
    out_shape=(25,25,16)
    oimg_sz=(100,100,1)
if img_size==(28,28):
    out_shape=(4,4,16)
    oimg_sz=(16,16,1)
#Sdata_path=input("Specify the folder where train and test folders are available:")
#"UTKFace_Age"
#"UTKFace_Age"
# Load the saved model from JSON file
data_path= input("Enter the Path of the images split into train and test folders with class labels as subfolders (Gal): ")
if data_path=="":
    data_path='Gal/'
if data_path.split()[-1] =='/':
    data_path='/'.join(data_path.split()[:-1])
# Load the data
# Load the data
def load_data(data_path,round_cnt=0):
    x = []
    fln=[]
    for root, dirs, files in os.walk(data_path):
        # Check if there are any files or subfolders in the current folder
        if len(files) > 0 or len(dirs) > 0:
            # Get the class name from the current folder name
            for file_name in files[round_cnt:round_cnt+mulfact*batch_size]:
                file_path = os.path.join(root, file_name)
                img = Image.open(file_path).convert('RGB')
                img = img.resize(img_size)
                img_arr = np.array(img)
                x.append(img_arr)
                fln.append(file_path)
    print("Reading %d to %d of %d"%(round_cnt,round_cnt+mulfact*batch_size,len(files)))
    x = np.array(x)
    return x,fln
round_cnt=0
while True:
    x_test, file_names = load_data(data_path,round_cnt)
    if len(file_names) <1:
        break

    num_layers=x_test.shape[3]
    print(num_layers)
    # Normalize the data
    x_test = x_test.astype('float32') / 255.
    ### Reading the architecture of the model and the saved model to predict on new data
    from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Dropout, Conv2D, Conv2DTranspose, BatchNormalization, Activation, MaxPooling2D, UpSampling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras import backend as K

    import tensorflow as tf
    from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Dropout, Attention, Lambda
    from tensorflow.keras.models import Model
    from tensorflow.keras import backend as K

    import tensorflow as tf
    from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Dropout, Attention, Lambda
    from tensorflow.keras.models import Model
    from tensorflow.keras import backend as K

    #autoencoder = autoencoder()
    # Load the model architecture
    with open('autoencoderHR_best.json', 'r') as json_file:
        loaded_model_json = json_file.read()

    autoencoder = tf.keras.models.model_from_json(loaded_model_json)
    if round_cnt==0:
        autoencoder.summary()
    else:
        print("--------------Round: %d ---------------"%round_cnt)
    #autoencoder.build(input_shape=(28,28,1))
    #autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    # Load the saved weights
    autoencoder.load_weights('autoencoderHR_best.h5')
    # Define the encoder and decoder functions using the loaded model
    encoder = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('max_pooling2d_2').output)
    decoder_input = tf.keras.Input(shape=out_shape)
    decoder = autoencoder.layers[-7](decoder_input)
    decoder = autoencoder.layers[-6](decoder)
    decoder = autoencoder.layers[-5](decoder)
    decoder = autoencoder.layers[-4](decoder)
    decoder = autoencoder.layers[-3](decoder)
    decoder = autoencoder.layers[-2](decoder)
    decoder = autoencoder.layers[-1](decoder)
    decoder = tf.keras.Model(inputs=decoder_input, outputs=decoder)

    ### Generating encoded and decoded images for all test samples and saving them
    #encoder = models.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('max_pooling2d_2').output)
    #decoder = autoencoder.decoder
    # Make predictions on the test data using the trained model
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)
    pth=os.path.split(data_path)[0].split('/')
    pth0='/'.join(pth[:-1])
    # Loop through all test samples
    for i in range(len(x_test)):
        # Get the class label of the current test sample
        file_name=os.path.basename(os.path.split(file_names[i])[1])
        pth1='/'.join(str(os.path.split(file_names[i])[0]).split('/')[-3:])
        # Create the directories for the predicted images
        #print(pth0,' ', pth1)
        if pth0 !='':

            os.makedirs('%s/Predict_encoded/%s/'%(pth0,pth1), exist_ok=True)
            os.makedirs('%s/Predict_decoder/%s/'%(pth0,pth1), exist_ok=True)
            os.makedirs('%s/Predict_encoded_arrays/%s/'%(pth0,pth1), exist_ok=True)
            encoded_filename0 = f'{pth0}/Predict_encoded_arrays/{pth1}/{file_name}.pkl'
            encoded_filename1 = f'{pth0}/Predict_encoded/{pth1}/{file_name}.png'
            decoded_filename = f'{pth0}/Predict_decoder/{pth1}/{file_name}.png'
        else:
            os.makedirs('Predict_encoded/%s'%pth1, exist_ok=True)
            os.makedirs('Predict_decoder/%s'%pth1, exist_ok=True)
            os.makedirs('Predict_encoded_arrays/%s'%pth1, exist_ok=True)
            encoded_filename0 = f'Predict_encoded_arrays/{pth1}/{file_name}.pkl'
            encoded_filename1 = f'Predict_encoded/{pth1}/{file_name}.png'
            decoded_filename = f'Predict_decoder/{pth1}/{file_name}.png'

        # Save the encoded output as a pkl file
        encddt=np.copy(encoded_imgs[i])
        encoded_output = encddt.flatten()
        #print(encoded_filename)
        with open(encoded_filename0, 'wb') as f:
            pickle.dump(encoded_output, f)

        # Save the encoded image
        encoded_img = encddt.reshape(oimg_sz)
        cv2.imwrite(encoded_filename1, encoded_img*255)

        # Save the decoded image
        img_array=np.array(np.uint8(decoded_imgs[i]*255))
        img_array =img_array.reshape(img_size[0],img_size[1],num_layers)
        # Convert the numpy array to a PIL image
        img = Image.fromarray(img_array)
        img.save(decoded_filename)
    round_cnt=round_cnt+mulfact*batch_size
