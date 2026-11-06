@'
from gradio_client import Client, handle_file
from pathlib import Path
import shutil
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SPACE_ID = "fashn-ai/fashn-vton-1.5"

def virtual_tryon(person_image, garment_image):
    print("Connecting to FASHN AI...", flush=True)

    client = Client(SPACE_ID)

    print("Generating AI try-on...", flush=True)

    result = client.predict(
        person_image=handle_file(str(person_image)),
        garment_image=handle_file(str(garment_image)),
        category="tops",
        garment_photo_type="flat-lay",
        num_timesteps=50,
        guidance_scale=1.5,
        seed=42,
        segmentation_free=True,
        api_name="/tryon"
    )

    if isinstance(result, dict):
        result_path = result.get("path")
        result_url = result.get("url")
    else:
        result_path = result
        result_url = None

    output_path = OUTPUT_DIR / "tryon_result.png"

    if result_path and Path(result_path).exists():
        shutil.copy2(result_path, output_path)
    elif result_url:
        response = requests.get(result_url, timeout=60)
        response.raise_for_status()
        output_path.write_bytes(response.content)
    else:
        raise RuntimeError(f"Could not retrieve result: {result}")

    print("SUCCESS! Result saved:", output_path, flush=True)

if __name__ == "__main__":
    person = BASE_DIR / "assets" / "person.jpeg"
    garment = BASE_DIR / "garments" / "blue_tshirt_transparent.png"

    print("Checking input images...", flush=True)

    if not person.exists():
        print("ERROR: assets/person.jpeg not found")
    elif not garment.exists():
        print("ERROR: Garment image not found")
    else:
        virtual_tryon(person, garment)
'@ | Set-Content -Encoding UTF8 backend\fashn_client.py