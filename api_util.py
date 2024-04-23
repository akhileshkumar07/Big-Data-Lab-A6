from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.requests import Request

import keras
import numpy as np
from PIL import Image

app = FastAPI()

# Function to load the Keras model from the specified path
def load_model(path: str) -> keras.Sequential:
    return keras.models.load_model(path)

# Function to predict the digit using the loaded model and input data
def predict_digit(model: keras.Sequential, data_point: list) -> str:
    # Reshape the data to match the input shape of the model
    data_point = np.array(data_point).reshape((1,784))
    # Make prediction
    prediction = model.predict(data_point)
    # Get the index of the digit with the highest probability
    digit = np.argmax(prediction)
    return str(digit)

# Function to format the uploaded image to match the input requirements of the model
def format_image(image: Image) -> list:
    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))
    # Convert the image to grayscale
    image = image.convert('L')
    # Normalize pixel values to range between 0 and 1
    data_point = [pixel / 255.0 for pixel in list(image.getdata())]
    return data_point

# Endpoint to handle POST requests for digit prediction
@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    # Open the uploaded image file
    image = Image.open(file.file)
    # Format the image for model input
    data_point = format_image(image)
    # Get the path to the model from app state
    model_path = request.app.state.model_path
    # Load the model
    model = load_model(model_path)
    # Predict the digit
    digit = predict_digit(model, data_point)
    # Return the prediction as JSON response
    return JSONResponse(content={"digit": digit}, media_type="application/json")