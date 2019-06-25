from flask import Flask, jsonify,request
from preprocessing.utils import get_video_frames, StreamWrapper
from preprocessing.transformer import ResNextTransformer
from preprocessing.spatial_transforms import validation_spatial_transform
import torch
from torchvision import transforms
import numpy as np
from model.model_zoo import ResnextClassifier
import os
import io

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
  stream = StreamWrapper(video.stream)
  frames = get_video_frames(stream)
  stream.close()

  transformer = ResNextTransformer(validation_spatial_transform, 64, 112)
  transformed_frames = transformer(frames)
  transformed_frames = torch.unsqueeze(transformed_frames, 0)
  predictions = np.array(model.predict(transformed_frames).squeeze())

  k = 5
  top_k = np.array(predictions).argsort()[-k:][::-1] # get top-k in descending order
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