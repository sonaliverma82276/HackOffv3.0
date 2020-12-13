from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import torch
from PIL import Image
import albumentations as aug


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

model = torch.load("hack.pth")
model.eval()
def model_predict(img_path, model):
    image = Image.open(img_path)
    image = np.array(image)
    transforms = aug.Compose([
            aug.Resize(384,384),
            aug.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225),max_pixel_value=255.0,always_apply=True),
            ])
    image = transforms(image=image)["image"]
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.tensor([image], dtype=torch.float)
    preds = model(image)
    preds = np.argmax(preds.detach())
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        labs=['AFIB','AFL','APB','Bigeminy','Fusion','IVR','LBBBB','NSR','PR','PVC','RBBBB','SDHB','SVTA','Trigeminy','VFL','VT','WPW']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        # Make prediction
        preds = model_predict(file_path, model)
        result = labs[preds]            
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
