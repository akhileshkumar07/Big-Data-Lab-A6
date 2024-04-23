**Handwritten Digit Recognition using Keras and FastAPI**
=====================================================

**Overview**
-----------

This repository contains a handwritten digit recognition system using Keras and FastAPI. The system consists of a Keras model trained on the MNIST dataset and a FastAPI application that accepts image uploads and returns the predicted digit.

**Getting Started**
---------------

### Prerequisites

* Python 3.7+
* Keras 2.4+
* FastAPI 0.65+
* Uvicorn 0.13+
* PIL (Python Imaging Library)

### Installation

1. Clone the repository: `git clone https://github.com/your-username/handwritten-digit-recognition.git`
2. Install the required packages: `pip install -r requirements.txt`
3. Train the model: `python train_model.py` (this will save the model to `mnist_model.keras`)

### Running the Application

1. Run the application: `python app.py /path/to/mnist_model.keras` (replace with the actual path to the saved model)
2. Open a web browser and navigate to `http://localhost:8000/docs` to access the API documentation
3. Use a tool like `curl` to test the endpoint by uploading an image file: `curl -X POST -F "file=@image.png" http://localhost:8000/predict`

**API Endpoint**
-------------

### `/predict`

* **Method**: POST
* **Request Body**: Image file (PNG or JPEG)
* **Response**: JSON object with the predicted digit (e.g., `{"digit": "5"}`)

**Model**
------

The Keras model is a sequential model with three dense layers: 256 units with ReLU activation, 128 units with ReLU activation, and 10 units with softmax activation. The model is trained on the MNIST dataset for 15 epochs with validation on the testing data.

**Acknowledgments**
---------------

This project uses the MNIST dataset, which is a widely used dataset for handwritten digit recognition. The dataset is available under the Creative Commons Attribution-Share Alike 3.0 license.
