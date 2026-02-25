import sys
import logging
import traceback
from datetime import timedelta, datetime
from pathlib import Path
import io
import os
import base64

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file

# ---------------- LOGGING (CRITICAL) ----------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ---------------- PROJECT IMPORTS ----------------
from inference import run_inference
import configs
from overlay import draw_head_analysis
from age_cal import compute_hc_mm_from_mask, ga_weeks_from_hc_intergrowth
from abnormality import (
    load_headcirc_table,
    cutoffs_exact_at_nearest_ga,
    percentile_band_from_cutoffs,
    classify_hc
)

app = Flask(__name__)

# ---------------- CONFIG ----------------
PROJECT_ROOT = Path(os.path.dirname(__file__))
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
EXCEL_PATH = PROJECT_ROOT / "data" / "FGCalculatorPercentileRange.xlsx"
TEMP_DIR = Path("/tmp/natalis_uploads")
TEMP_DIR.mkdir(exist_ok=True)

# ---------------- LOAD TABLE ----------------
try:
    HC_TABLE = load_headcirc_table(str(EXCEL_PATH))
    MODEL_LOADED = True
    logging.info("✅ HC table loaded successfully")
except Exception:
    logging.exception("❌ Failed to load HC table")
    HC_TABLE = None
    MODEL_LOADED = False


# ---------------- CLINICAL CAL ----------------
def calculate_edd_info(ga_weeks, scan_date_str):
    if ga_weeks is None:
        {"edd": None, "trimester": "N/A", "weeks_remaining": None}

    scan_date = datetime.strptime(scan_date_str, "%Y-%m-%d")
    conception_date = scan_date - timedelta(days=int(ga_weeks * 7))
    edd = conception_date + timedelta(days=280)

    remaining_days = (edd - scan_date).days
    weeks = remaining_days // 7
    days = remaining_days % 7

    trimester = (
        "First Trimester" if ga_weeks < 13 else
        "Second Trimester" if ga_weeks < 27 else
        "Third Trimester"
    )

    return {
        "edd": edd.strftime("%Y-%m-%d"),
        "trimester": trimester,
        "weeks_remaining": f"{weeks} weeks {days} days"
    }


# ==================== API ====================

@app.route("/api/analyze_image", methods=["POST"])
def analyze_image():
    try:
        logging.info("📥 /api/analyze_image request received")

        if not MODEL_LOADED:
            return jsonify({"error": "HC table not loaded"}), 500

        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        race = request.form.get("race", configs.ALLOWED_RACES[0])
        scan_date_str = request.form.get("scan_date", datetime.now().strftime("%Y-%m-%d"))

        try:
            pixel_size_mm = float(request.form.get("pixel_size_mm", configs.PIXEL_SIZE_MM))
        except ValueError:
            return jsonify({"error": "Invalid pixel_size_mm"}), 400

        # ---------- LOAD IMAGE ----------
        image_file = request.files["image"]
        img_pil = Image.open(io.BytesIO(image_file.read())).convert("L")

        temp_input_path = TEMP_DIR / f"{os.getpid()}_{datetime.now().timestamp()}.png"
        img_pil.save(temp_input_path)

        logging.info("🖼️ Image saved, running inference...")

        # ---------- INFERENCE ----------
        result = run_inference(str(MODEL_PATH), str(temp_input_path))

        if result is None or not isinstance(result, dict):
            raise RuntimeError("run_inference returned invalid result")

        if "prediction" not in result or "input_image" not in result:
            raise KeyError(f"run_inference keys missing: {result.keys()}")

        prediction = result["prediction"]
        input_image_np = result["input_image"]

        if prediction is None or np.count_nonzero(prediction) == 0:
            return jsonify({"error": "Empty segmentation mask"}), 422

        logging.info("✅ Inference successful")

        # ---------- CLEANUP ----------
        os.remove(temp_input_path)

        # ---------- BIOMETRY ----------
        hc_mm = compute_hc_mm_from_mask(prediction, pixel_size_mm)

        if hc_mm is None or hc_mm <= 0:
            return jsonify({"error": "Invalid HC measurement"}), 422

        ga_weeks = ga_weeks_from_hc_intergrowth(hc_mm)

        if ga_weeks is None:
            percentile_band = "N/A"
            classification = "N/A"
            ga_used = None
        else:
            cut_mm, ga_used = cutoffs_exact_at_nearest_ga(
                HC_TABLE, race=race, ga_weeks=ga_weeks
            )
            percentile_band = percentile_band_from_cutoffs(hc_mm, cut_mm)
            classification = classify_hc(hc_mm, cut_mm)

        clinical_info = calculate_edd_info(ga_weeks, scan_date_str)

        # ---------- ANNOTATION ----------
        annotated_img_np = draw_head_analysis(
            input_image_np, prediction, hc_mm, pixel_size_mm
        )

        annotated_pil = Image.fromarray(annotated_img_np.astype(np.uint8))
        temp_output_path = TEMP_DIR / f"{os.getpid()}_{datetime.now().timestamp()}_annotated.png"
        annotated_pil.save(temp_output_path)

        logging.info("🧠 Analysis complete")

        return jsonify({
            "status": "success",
            "hc_mm": round(hc_mm, 2),
            "ga_weeks": round(ga_weeks, 2) if ga_weeks else None,
            "classification": classification,
            "percentile_band": percentile_band,
            "edd": clinical_info["edd"],
            "trimester": clinical_info["trimester"],
            "weeks_remaining": clinical_info["weeks_remaining"],
            "race": race,
            "scan_date": scan_date_str,
            "pixel_size_mm": pixel_size_mm,
            "annotated_image_id": temp_output_path.name
        })

    except Exception:
        app.logger.exception("🔥 UNHANDLED ERROR in /api/analyze_image")
        return jsonify({"error": "Internal server error"}), 500


# ---------- IMAGE RETRIEVAL ----------
@app.route("/api/get_annotated_image/<image_id>", methods=["GET"])
def get_annotated_image(image_id):
    """
    Retrieves the previously generated annotated image by its ID
    and returns it as a Base64-encoded string in JSON.
    """
    image_path = TEMP_DIR / image_id

    if not image_path.exists():
        return jsonify({"error": "Annotated image not found"}), 404

    try:
        with open(image_path, "rb") as f:
            encoded_bytes = base64.b64encode(f.read())
            encoded_str = encoded_bytes.decode("utf-8")
            # Wrap in data URI format for direct frontend use
            data_uri = f"data:image/png;base64,{encoded_str}"

        return jsonify({
            "status": "success",
            "image_id": image_id,
            "image_base64": data_uri
        })

    except Exception as e:
        app.logger.exception("🔥 Error encoding annotated image to Base64")
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5003,
        use_reloader=False  # IMPORTANT
    )