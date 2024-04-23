import uvicorn
import sys

# Import the FastAPI app instance from api_util module
from api_util import app

# Get the model path from command line arguments
model_path = sys.argv[1]

# Set the model path in the app state for later use in the API endpoints
app.state.model_path = model_path

# Run the FastAPI app using Uvicorn server
uvicorn.run(app, host="127.0.0.1", port=8000)