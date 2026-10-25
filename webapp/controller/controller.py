"""Flask web app: serves the UI and forwards classification requests.

No PyTorch here — all model work is delegated to the inference service.
"""
from flask import Flask, jsonify, render_template, request

from services.inference_client import InferenceClient, InferenceError

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB (Flask rejects larger bodies with 413)

inference = InferenceClient()

controller = Flask(__name__,
                   template_folder="../view/templates",
                   static_folder="../view/static")
controller.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@controller.route("/")
def index():
    try:
        models = inference.models()
    except InferenceError:
        models = {}  # page still renders; classify will report the error
    return render_template("index.html", models=models)


@controller.route("/classify", methods=["POST"])
def classify_image():
    file = request.files.get("image")
    model_id = request.form.get("model_id", "vgg")

    if file is None:
        return jsonify({"error": "No se recibió ninguna imagen."}), 400
    if file.filename == "":
        return jsonify({"error": "El nombre de archivo está vacío."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Formato de imagen no soportado."}), 400

    try:
        # Read into memory and forward — nothing is written to disk.
        result = inference.classify(file.read(), file.filename, file.mimetype, model_id)
        return jsonify(result)
    except InferenceError as e:
        return jsonify({"error": f"Error durante la clasificación: {e}"}), e.status


@controller.errorhandler(413)
def too_large(_):
    return jsonify({"error": "La imagen supera el límite de 10 MB."}), 413
