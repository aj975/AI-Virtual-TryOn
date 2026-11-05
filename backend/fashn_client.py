
import os
import shutil
import requests

from gradio_client import Client, handle_file
from pathlib import Path
from dotenv import load_dotenv

# =====================================
# PROJECT PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Load Hugging Face token from project .env
load_dotenv(BASE_DIR / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SPACE_ID = "fashn-ai/fashn-vton-1.5"


# =====================================
# AI VIRTUAL TRY-ON FUNCTION
# =====================================

def virtual_tryon(person_image, garment_image):

    print("\nConnecting to FASHN AI...", flush=True)

    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN not found. Check your project .env file."
        )

    # Authenticate using Hugging Face token
    client = Client(SPACE_ID, token=HF_TOKEN)

    print("Uploading images...", flush=True)
    print("Generating your AI try-on...", flush=True)

    result = client.predict(
        person_image=handle_file(str(person_image)),
        garment_image=handle_file(str(garment_image)),
        category="tops",
        garment_photo_type="flat-lay",
        num_timesteps=50,
        guidance_scale=1.5,
        seed=42,
        segmentation_free=True,
        api_name="/try_on"
    )

    # =====================================
    # RETRIEVE GENERATED IMAGE
    # =====================================

    output_path = OUTPUT_DIR / "tryon_result.png"

    if isinstance(result, dict):

        result_path = result.get("path")
        result_url = result.get("url")

    else:

        result_path = result
        result_url = None

    if result_path and Path(result_path).exists():

        shutil.copy2(result_path, output_path)

    elif result_url:

        response = requests.get(result_url, timeout=60)
        response.raise_for_status()

        output_path.write_bytes(response.content)

    else:

        raise RuntimeError(
            f"Could not retrieve generated image: {result}"
        )

    print("\nSUCCESS! AI TRY-ON COMPLETED!", flush=True)
    print("Your result is saved at:", output_path, flush=True)

    return output_path


# =====================================
# MAIN PROGRAM
# =====================================

if __name__ == "__main__":

    person = BASE_DIR / "assets" / "person.jpeg"

    garment = (
        BASE_DIR
        / "garments"
        / "blue_tshirt_transparent.png"
    )

    print("\nChecking images...", flush=True)

    if not person.exists():

        print("ERROR: Add your photo as assets/person.jpeg")

    elif not garment.exists():

        print("ERROR: Garment image not found")

    else:

        virtual_tryon(person, garment)
