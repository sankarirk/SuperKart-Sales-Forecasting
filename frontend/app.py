import os
import requests
import streamlit as st


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="SuperKart Sales Prediction",
    page_icon="🛒",
    layout="centered"
)


# ============================================================
# Backend configuration
# ============================================================

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:7860"
).rstrip("/")

PREDICTION_URL = f"{BACKEND_URL}/v1/predict"


# ============================================================
# Page header
# ============================================================

st.title("🛒 SuperKart Sales Prediction")

st.markdown(
    """
    Predict the expected sales revenue of a SuperKart product
    using the trained machine learning model.
    """
)

st.divider()


# ============================================================
# Prediction form
# ============================================================

st.subheader("Enter Product & Store Details")

with st.form("prediction_form"):

    # --------------------------------------------------------
    # Product information
    # --------------------------------------------------------

    st.markdown("### Product Information")

    product_weight = st.number_input(
        "Product Weight",
        min_value=0.0,
        value=9.3,
        step=0.1,
        format="%.2f"
    )

    product_sugar_content = st.selectbox(
        "Product Sugar Content",
        [
            "Low Sugar",
            "No Sugar",
            "Regular"
        ]
    )

    product_allocated_area = st.number_input(
        "Product Allocated Area",
        min_value=0.0,
        value=0.016,
        step=0.001,
        format="%.3f"
    )

    product_mrp = st.number_input(
        "Product MRP",
        min_value=0.0,
        value=249.8,
        step=1.0,
        format="%.2f"
    )

    product_category = st.selectbox(
        "Product Category",
        [
            "Perishable",
            "Non-Perishable"
        ]
    )

    product_family = st.selectbox(
        "Product Family",
        [
            "FD",
            "DR",
            "NC"
        ]
    )

    # --------------------------------------------------------
    # Store information
    # --------------------------------------------------------

    st.markdown("### Store Information")

    store_size = st.selectbox(
        "Store Size",
        [
            "Small",
            "Medium",
            "High"
        ]
    )

    store_location_city_type = st.selectbox(
        "Store Location City Type",
        [
            "Tier 1",
            "Tier 2",
            "Tier 3"
        ]
    )

    store_type = st.selectbox(
        "Store Type",
        [
            "Departmental Store",
            "Food Mart",
            "Supermarket Type1",
            "Supermarket Type2"
        ]
    )

    store_age_years = st.number_input(
        "Store Age (Years)",
        min_value=0,
        value=10,
        step=1
    )

    # --------------------------------------------------------
    # Submit
    # --------------------------------------------------------

    submitted = st.form_submit_button(
        "🔮 Predict Sales",
        use_container_width=True
    )


# ============================================================
# Prediction
# ============================================================

if submitted:

    payload = {

        "Product_Weight":
            product_weight,

        "Product_Sugar_Content":
            product_sugar_content,

        "Product_Allocated_Area":
            product_allocated_area,

        "Product_MRP":
            product_mrp,

        "Store_Size":
            store_size,

        "Store_Location_City_Type":
            store_location_city_type,

        "Store_Type":
            store_type,

        "Store_Age_Years":
            store_age_years,

        "Product_Category":
            product_category,

        "Product_Family":
            product_family

    }

    with st.spinner("Generating prediction..."):

        try:

            response = requests.post(
                PREDICTION_URL,
                json=payload,
                timeout=30
            )

            # ------------------------------------------------
            # Successful prediction
            # ------------------------------------------------

            if response.status_code == 200:

                result = response.json()

                predicted_sales = result.get(
                    "predicted_sales"
                )

                if predicted_sales is not None:

                    st.success(
                        "Prediction generated successfully!"
                    )

                    st.metric(
                        label="Predicted Sales",
                        value=f"{predicted_sales:,.2f}"
                    )

                else:

                    st.error(
                        "The backend returned an unexpected response."
                    )

            # ------------------------------------------------
            # Client-side validation error
            # ------------------------------------------------

            elif response.status_code == 400:

                result = response.json()

                st.error(
                    result.get(
                        "message",
                        "Invalid input."
                    )
                )

                details = result.get("details")

                if details:

                    st.write("Details:", details)

            # ------------------------------------------------
            # Server error
            # ------------------------------------------------

            else:

                try:
                    result = response.json()

                    error_message = result.get(
                        "message",
                        "Backend server error."
                    )

                    st.error(error_message)

                    if result.get("details"):
                        st.write(
                            "Details:",
                            result["details"]
                        )

                except Exception:

                    st.error(
                        f"Backend returned HTTP "
                        f"{response.status_code}."
                    )

        except requests.exceptions.ConnectionError:

            st.error(
                "Unable to connect to the prediction backend."
            )

            st.info(
                f"Backend URL: {BACKEND_URL}"
            )

        except requests.exceptions.Timeout:

            st.error(
                "The prediction request timed out."
            )

        except requests.exceptions.RequestException as error:

            st.error(
                f"Request failed: {error}"
            )

        except Exception as error:

            st.error(
                f"Unexpected error: {error}"
            )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "SuperKart Sales Forecasting • "
    "Machine Learning Prediction Application"
)