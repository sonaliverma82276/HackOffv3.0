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
from efficientnet_pytorch import EfficientNet

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# model = torch.load("hack909.pth")
model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=17)
model.eval()
def model_predict(file, model):
    image = Image.open(file)
    image = np.array(image)
    transforms = aug.Compose([
            aug.Resize(224,224),
            aug.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225),max_pixel_value=255.0,always_apply=True),
            ])
    image = transforms(image=image)["image"]
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.tensor([image], dtype=torch.float)
    preds = model(image)
    preds = np.argmax(preds.detach())
    return preds


@app.route('/')
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    # Get the file from post request
    f = request.files['file']
    labs=['AFIB','AFL','APB','Bigeminy','Fusion','IVR','LBBBB','NSR','PR','PVC','RBBBB','SDHB','SVTA','Trigeminy','VFL','VT','WPW']

    # Make prediction
    preds = model_predict(f, model)
    result = labs[preds]            
    return result


if __name__ == '__main__':
    app.run(debug=True)
