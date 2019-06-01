from flask import Flask, jsonify,request
from preprocessing.utils import get_video_frames
from preprocessing.transformer import ResNextTransformer
from preprocessing.spatial_transforms import validation_spatial_transform
import torch
from torchvision import transforms
import numpy as np
from model.model_zoo import ResnextClassifier
import os
import matplotlib.image as mpimg

app = Flask(__name__)

model = ResnextClassifier('best_params_100_deep_64.pth')

@app.after_request
def allow_CORS(response):
  response.headers["Access-Control-Allow-Origin"] = "*"
  return response

@app.route("/")
def home():
  return "home"

@app.route("/predict", methods=['POST'])
def predict():
  video = request.files['video']
  file_path = 'video.avi'
  video.save(file_path)
  frames = get_video_frames(file_path)
  if os.path.isfile(file_path):
    os.remove(file_path)

  transformer = ResNextTransformer(validation_spatial_transform, 64, 112)
  transformed_frames = transformer(frames)
  transformed_frames = torch.unsqueeze(transformed_frames, 0)
  predictions = np.array(model.predict(transformed_frames).squeeze())

  k = 5
  top_k = np.array(predictions).argsort()[-k:][::-1]
  res = {
    "predictions": [{
      "id": str(id),
      "probability": str(prob)
      } for id, prob in zip(top_k, predictions[top_k])]
    }
  print(res)
  return jsonify(res)

if __name__ == "main":
  app.run()