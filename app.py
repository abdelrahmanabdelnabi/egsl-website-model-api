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
from class_names import class_names
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model_path = os.environ['MODEL_PATH']

model = ResnextClassifier(model_path, n_classes=len(class_names))

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
      "word": class_names[id],
      "probability": str(prob)
      } for id, prob in zip(top_k, predictions[top_k])]
    }
  print(res)
  return jsonify(res)

if __name__ == "main":
  app.run()