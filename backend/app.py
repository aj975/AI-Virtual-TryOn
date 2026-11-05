from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pathlib import Path
from uuid import uuid4
import base64
import binascii

from fashn_client import virtual_tryon

app = Flask(__name__)
CORS(app, origins=["http://localhost:8000", "http://127.0.0.1:8000"])

BASE = Path(__file__).resolve().parent.parent
OUTPUTS = BASE / "outputs"
OUTPUTS.mkdir(exist_ok=True)

GARMENT = BASE / "garments" / "blue_tshirt.webp"


@app.route("/api/health")
def health():
    return jsonify({"status": "running", "message": "AI backend ready"})


@app.route("/api/tryon", methods=["POST"])
def tryon():
    try:
        data = request.get_json(silent=True) or {}
        photo = data.get("photo")

        if not isinstance(photo, str) or not photo.startswith("data:image/png;base64,"):
            return jsonify({"error": "A captured PNG photo is required"}), 400

        if not GARMENT.is_file():
            return jsonify({"error": "Garment image not found"}), 500

        encoded = photo.split(",", 1)[1]

        if len(encoded) > 20_000_000:
            return jsonify({"error": "Photo is too large"}), 413

        image_bytes = base64.b64decode(encoded, validate=True)

        job_id = uuid4().hex
        photo_path = OUTPUTS / f"capture_{job_id}.png"
        photo_path.write_bytes(image_bytes)

        print("Generating AI try-on...", flush=True)
        result_path = virtual_tryon(photo_path, GARMENT)

        if not result_path:
            raise RuntimeError("FASHN did not return an image path")

        result_path = Path(result_path)

        if not result_path.is_file():
            raise RuntimeError("Generated image was not found")

        final_path = OUTPUTS / f"web_tryon_{job_id}.png"

        import shutil
        shutil.copyfile(result_path, final_path)

        return jsonify({
            "success": True,
            "image_url": f"http://localhost:5000/api/result/{final_path.name}"
        })

    except (ValueError, binascii.Error) as error:
        return jsonify({"error": str(error)}), 400

    except Exception as error:
        print("TRY-ON ERROR:", error, flush=True)
        return jsonify({"error": str(error)}), 500


@app.route("/api/result/<filename>")
def result(filename):
    path = OUTPUTS / filename

    if not filename.startswith("web_tryon_") or not path.is_file():
        return jsonify({"error": "Result not found"}), 404

    return send_file(path, mimetype="image/png")


if __name__ == "__main__":
    print("AI backend starting at http://localhost:5000", flush=True)
    app.run(host="127.0.0.1", port=5000, debug=False)
