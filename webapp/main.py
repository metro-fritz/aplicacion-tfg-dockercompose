# Local development entry point (`python main.py`).
# In Docker the app is served by gunicorn instead (see Dockerfile).
from controller.controller import controller

if __name__ == "__main__":
    controller.run(host="0.0.0.0", port=5000, debug=False)
