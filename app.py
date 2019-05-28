from flask import Flask, jsonify,request

app = Flask(__name__)

@app.route("/")
def home():
  return "home"

@app.route("/predict", methods=['POST'])
def predict():
  print('request to predict')
  video = request.files['video']
  video.save('video.mp4')
  return jsonify({ "prediction": "1" })

if __name__ == "main":
  app.run(debug=True)