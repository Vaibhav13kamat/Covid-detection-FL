from flask import Flask, render_template
from PIL import Image
import numpy as np
import io
import os


app = Flask(__name__)

@app.route('/')
def index():
    image_names = os.listdir('dataset_split/client1/train/Normal')[:5] # Replace 'normal' with your class name
    return render_template('index.html', image_names=image_names)

@app.route('/dataset_split/client1/train/Normal/<image_name>')
def serve_image(image_name):
    filename = os.path.join('dataset_split/client1/train/Normal', image_name) # Replace 'normal' with your class name
    image = Image.open(filename)   #.convert('RGB').resize((224, 224))
    # image_array = np.array(image) / 255.0
    # return image_array.tobytes()
    image = image.resize((224, 224))
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    return img_bytes.getvalue()

if __name__ == '__main__':
    app.run(port=9970, debug=True)
