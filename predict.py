from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
import numpy as np
import tensorflow as tf
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator







def pred(image1):
    weights1 = np.load('round-3-weights.npz')
    # # Construct the model architecture
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    #     tf.keras.layers.Dense(10, activation='softmax')
    # ])

    # # Compile the model
    # model.compile(optimizer='adam',
    #             loss='sparse_categorical_crossentropy',
    #             metrics=['accuracy'])
    
    # Load and compile Keras model
    vgg = VGG19(weights=weights1, include_top=False, input_shape=(112, 112, 3))
    # Freeze first 10 layers
    for layer in vgg.layers[:10]:
        layer.trainable = False
    x = vgg.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)  # change number of classes to 2 for covid and normal
    model = Model(inputs=vgg.input, outputs=predictions)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])





    
    # Set the model weights
    model.set_weights([weights1[name] for name in weights1])
    # Make a prediction on new input data
    # x_test = np.random.randn(100, 784)
    # y_pred = model.predict(x_test)

    x_test = preprocess_image(image1)

    # Call the predict function with the preprocessed image
    y_pred = model.predict(x_test)

    return y_pred


def preprocess_image(image_path):
    # Load the image
    img = Image.open(image_path)

    # Resize the image to (112, 112)
    img = img.resize((112, 112))

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Add an extra dimension to the image array for batching
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the image array
    img_array = img_array.astype('float32') / 255.0

    return img_array



















app = Flask(__name__)

# Set Tesseract path for OCR
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

@app.route('/api/extract_text', methods=['POST'])
def extract_text():
    # Check if image file is present in the request
    if 'image' not in request.files:
        return jsonify({'error': 'no image found in request'}), 400

    # Read image file and convert to grayscale
    image_file = request.files['image']
    image = Image.open(image_file).convert('L')

    # Extract text from image using pytesseract
    # text = pytesseract.image_to_string(image)

    x=pred(image_file)
    result="positive"
    accuracy="99%"
    # Return text as response
    return jsonify({'result': x,
                    'accuracy':accuracy})

if __name__ == '__main__':
    app.run(debug=True)


#commandline http request
#curl -X POST -F "image=@ocr_test.png" http://localhost:5000/api/extract_text

