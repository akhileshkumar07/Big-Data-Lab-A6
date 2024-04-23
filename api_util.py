from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.requests import Request

import keras
import numpy as np
from PIL import Image

app = FastAPI()

def load_model(path: str) -> keras.Sequential:
    return keras.models.load_model(path)

def predict_digit(model: keras.Sequential, data_point: list) -> str:
    data_point = np.array(data_point).reshape((1,784))
    prediction = model.predict(data_point)
    digit = np.argmax(prediction)
    return str(digit)

def format_image(image: Image) -> list:
    image = image.resize((28, 28))
    image = image.convert('L')  # Convert to grayscale
    data_point = [pixel / 255.0 for pixel in list(image.getdata())]
    return data_point

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    image = Image.open(file.file)
    data_point = format_image(image)
    model_path = request.app.state.model_path
    model = load_model(model_path)
    digit = predict_digit(model, data_point)
    return JSONResponse(content={"digit": digit}, media_type="application/json")