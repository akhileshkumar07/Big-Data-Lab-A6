import uvicorn
import sys

from api_util import app

model_path = sys.argv[1]
app.state.model_path = model_path

uvicorn.run(app, host="127.0.0.1", port=8000)