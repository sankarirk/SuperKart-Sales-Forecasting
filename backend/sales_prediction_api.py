import os
import logging

import joblib
import pandas as pd

from flask import Flask, jsonify, request


# ============================================================
# Application setup
# ============================================================

app = Flask(__name__)

app.config["JSON_SORT_KEYS"] = False


# ============================================================
# Logging configuration
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("superkart_api")


# ============================================================
# Model configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = "superkart_sales_pipeline.joblib"

MODEL_PATH = os.path.join(
    BASE_DIR,
    MODEL_FILE
)


# ============================================================
# Load trained model
# ============================================================

try:

    model = joblib.load(MODEL_PATH)

    logger.info(
        "Model loaded successfully from %s",
        MODEL_PATH
    )

except FileNotFoundError:

    logger.exception(
        "Model file was not found: %s",
        MODEL_PATH
    )

    raise

except Exception:

    logger.exception(
        "Unexpected error while loading the model."
    )

    raise


# ============================================================
# Required model features
# ============================================================

REQUIRED_FEATURES = [

    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Store_Age_Years",
    "Product_Type_Category",
    "Product_Id_char"

]


# ============================================================
# Input validation helper
# ============================================================

def validate_payload(payload):
    """
    Check whether the request contains all
    features required by the trained pipeline.
    """

    if payload is None:

        return [
            "Request body must contain valid JSON."
        ]

    missing_features = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in payload
    ]

    return missing_features


# ============================================================
# Convert request data into model-ready DataFrame
# ============================================================

def prepare_input(payload):
    """
    Convert the incoming JSON record into a
    pandas DataFrame with the same feature
    names expected by the trained pipeline.
    """

    record = {

        "Product_Weight": float(
            payload["Product_Weight"]
        ),

        "Product_Sugar_Content":
            str(payload["Product_Sugar_Content"]),

        "Product_Allocated_Area": float(
            payload["Product_Allocated_Area"]
        ),

        "Product_MRP": float(
            payload["Product_MRP"]
        ),

        "Store_Size":
            str(payload["Store_Size"]),

        "Store_Location_City_Type":
            str(payload["Store_Location_City_Type"]),

        "Store_Type":
            str(payload["Store_Type"]),

        "Store_Age_Years": int(
            payload["Store_Age_Years"]
        ),

        "Product_Type_Category":
            str(payload["Product_Type_Category"]),

        "Product_Id_char":
            str(payload["Product_Id_char"])

    }

    return pd.DataFrame([record])


# ============================================================
# Health-check endpoint
# ============================================================

@app.route("/health", methods=["GET"])
def health_check():

    return jsonify({

        "status": "healthy",

        "service":
            "SuperKart Sales Prediction API",

        "model_loaded":
            model is not None

    })


# ============================================================
# API information endpoint
# ============================================================

@app.route("/", methods=["GET"])
def api_information():

    return jsonify({

        "application":
            "SuperKart Sales Prediction API",

        "version":
            "1.0",

        "description":
            "REST API for predicting SuperKart product sales.",

        "endpoints": {

            "health":
                "GET /health",

            "prediction":
                "POST /v1/predict"

        },

        "required_features":
            REQUIRED_FEATURES

    })


# ============================================================
# Prediction endpoint
# ============================================================

@app.route("/v1/predict", methods=["POST"])
def predict_sales():

    logger.info(
        "Prediction request received."
    )

    try:

        payload = request.get_json(
            silent=True
        )

        missing_features = validate_payload(
            payload
        )

        if missing_features:

            logger.warning(
                "Invalid request: %s",
                missing_features
            )

            return jsonify({

                "status": "error",

                "message":
                    "Invalid prediction request.",

                "details":
                    missing_features

            }), 400


        input_df = prepare_input(
            payload
        )


        logger.info(
            "Input successfully prepared for prediction."
        )


        prediction = model.predict(
            input_df
        )[0]


        logger.info(
            "Prediction completed successfully."
        )


        return jsonify({

            "status": "success",

            "predicted_sales":
                round(float(prediction), 2)

        })


    except (ValueError, TypeError) as error:

        logger.warning(
            "Invalid input values: %s",
            error
        )

        return jsonify({

            "status": "error",

            "message":
                "One or more input values are invalid.",

            "details":
                str(error)

        }), 400


    except Exception as error:

        logger.exception(
            "Prediction failed unexpectedly."
        )

        return jsonify({

            "status": "error",

            "message":
                "Unable to generate prediction.",

            "details":
                str(error)

        }), 500


# ============================================================
# Application entry point
# ============================================================

if __name__ == "__main__":

    logger.info(
        "Starting SuperKart Sales Prediction API."
    )

    app.run(
        host="0.0.0.0",
        port=7860,
        debug=False
    )