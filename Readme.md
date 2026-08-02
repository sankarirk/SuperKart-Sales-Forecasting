

````markdown
# 🛒 SuperKart Sales Forecasting

An end-to-end Machine Learning application for predicting product sales revenue for SuperKart stores.

The project covers the complete machine learning lifecycle, from data analysis and feature engineering to model training, API development, frontend development, Docker containerization, and cloud deployment.

---

## 🌐 Live Application

### Frontend

🚀 **Live App:**  
https://superkart-sales-forecasting-1.onrender.com

The Streamlit frontend allows users to enter product and store information and receive a predicted sales value.

### Backend API

🔌 **Backend API:**  
https://superkart-sales-forecasting.onrender.com

### API Health Check

https://superkart-sales-forecasting.onrender.com/health

### GitHub Repository

https://github.com/sankarirk/SuperKart-Sales-Forecasting

---

# 📌 Project Overview

SuperKart is a retail organization operating multiple stores across different locations.

The objective of this project is to build a machine learning model that predicts the expected sales revenue of a product based on product and store characteristics.

Accurate sales forecasting can help businesses with:

- Inventory planning
- Revenue forecasting
- Product assortment optimization
- Store-level decision making
- Demand analysis
- Regional sales planning

The final solution uses an optimized **Random Forest Regression** model and exposes the trained model through a Flask REST API.

A Streamlit frontend provides a user-friendly interface for generating predictions.

---

# 🏗️ System Architecture

```text
                         ┌──────────────────────┐
                         │        User          │
                         │                      │
                         │ Product & Store Data │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Streamlit Frontend   │
                         │                      │
                         │     Render Cloud     │
                         └──────────┬───────────┘
                                    │
                                    │ HTTP POST
                                    ▼
                    ┌───────────────────────────────┐
                    │       Flask REST API          │
                    │                               │
                    │       /health                 │
                    │       /v1/predict             │
                    └──────────────┬────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────┐
                    │ Scikit-learn Pipeline         │
                    │                               │
                    │ StandardScaler                │
                    │ OneHotEncoder                 │
                    │ RandomForestRegressor          │
                    └──────────────┬────────────────┘
                                   │
                                   ▼
                         ┌──────────────────────┐
                         │ Predicted Sales      │
                         │                      │
                         │ Revenue Prediction   │
                         └──────────────────────┘
````

---

# 📂 Project Structure

```text
SuperKart-Sales-Forecasting/
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── sales_prediction_api.py
│   └── superkart_sales_pipeline.joblib
│
├── frontend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py
│
├── notebook/
│   └── SupertKart.ipynb
│
├── data/
│   └── ...
│
├── README.md
└── ...
```

---

# 🧠 Machine Learning Workflow

The project follows an end-to-end machine learning workflow:

```text
Raw Dataset
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Data Cleaning
     │
     ▼
Feature Engineering
     │
     ▼
Feature Selection
     │
     ▼
Train / Test Split
     │
     ▼
Data Preprocessing
     │
     ├── StandardScaler
     │
     └── OneHotEncoder
     │
     ▼
Model Training
     │
     ▼
Hyperparameter Optimization
     │
     ▼
Model Evaluation
     │
     ▼
Optimized Random Forest
     │
     ▼
Model Serialization
     │
     ▼
Flask REST API
     │
     ▼
Streamlit Frontend
     │
     ▼
Cloud Deployment
```

---

# 🔧 Feature Engineering

## 1. Product Category

A new feature called `Product_Category` was created to classify products as:

* `Perishable`
* `Non-Perishable`

The following product types were considered perishable:

```text
Dairy
Meat
Fruits and Vegetables
Breakfast
Breads
Seafood
```

The feature was created using:

```python
perishable_items = [
    "Dairy",
    "Meat",
    "Fruits and Vegetables",
    "Breakfast",
    "Breads",
    "Seafood"
]

data["Product_Category"] = np.where(
    data["Product_Type"].isin(perishable_items),
    "Perishable",
    "Non-Perishable"
)
```

---

## 2. Product Family

The first two characters of `Product_Id` were extracted to create a new feature called `Product_Family`.

Example:

```text
FD12345 → FD
DR12345 → DR
NC12345 → NC
```

The resulting categories are:

```text
FD
DR
NC
```

---

## 3. Store Age

`Store_Establishment_Year` was transformed into:

```text
Store_Age_Years
```

This provides a more meaningful numerical representation of store age for the machine learning model.

---

# 🗑️ Removed Features

The following columns were removed after feature engineering:

```text
Product_Id
Store_Id
Product_Type
Store_Establishment_Year
```

### Reasons

**Product_Id**

Used to create `Product_Family`.

**Product_Type**

Used to create `Product_Category`.

**Store_Establishment_Year**

Transformed into `Store_Age_Years`.

**Store_Id**

Considered an identifier rather than a useful predictive feature.

---

# 📊 Final Model Features

The final machine learning model uses the following 10 features:

```text
Product_Weight
Product_Sugar_Content
Product_Allocated_Area
Product_MRP
Store_Size
Store_Location_City_Type
Store_Type
Store_Age_Years
Product_Category
Product_Family
```

---

# ⚙️ Data Preprocessing

The final model is stored as a Scikit-learn Pipeline.

## Numerical Features

The following numerical features are standardized using `StandardScaler`:

```text
Product_Weight
Product_Allocated_Area
Product_MRP
Store_Age_Years
```

## Categorical Features

The following categorical features are encoded using `OneHotEncoder`:

```text
Product_Sugar_Content
Store_Size
Store_Location_City_Type
Store_Type
Product_Category
Product_Family
```

The encoder uses:

```python
OneHotEncoder(handle_unknown="ignore")
```

This allows the model to handle unknown categorical values without failing during prediction.

---

# 🤖 Machine Learning Model

Multiple regression models were evaluated during the project, including:

* Random Forest
* XGBoost

Hyperparameter optimization was performed using `GridSearchCV`.

The optimized **Random Forest Regressor** was selected as the final production model.

### Model Performance

| Metric   |                   Score |
| -------- | ----------------------: |
| R² Score |                  0.9317 |
| Model    | Random Forest Regressor |

The model achieved an R² score of approximately **0.9317** on unseen test data.

---

# 💾 Model Serialization

The complete trained Scikit-learn pipeline was serialized using `joblib`.

Model file:

```text
backend/superkart_sales_pipeline.joblib
```

The saved pipeline contains:

```text
Input Features
      │
      ▼
ColumnTransformer
      │
      ├── StandardScaler
      │
      └── OneHotEncoder
      │
      ▼
RandomForestRegressor
      │
      ▼
Prediction
```

Because the complete preprocessing pipeline is saved with the model, the API can directly accept raw feature values.

---

# 🔌 REST API

The backend is implemented using:

* Flask
* Gunicorn
* Pandas
* Scikit-learn
* Joblib

## Production Backend

```text
https://superkart-sales-forecasting.onrender.com
```

---

# ❤️ Health Check

### Endpoint

```text
GET /health
```

Example:

```bash
curl https://superkart-sales-forecasting.onrender.com/health
```

Expected response:

```json
{
  "model_loaded": true,
  "service": "SuperKart Sales Prediction API",
  "status": "healthy"
}
```

---

# 🔮 Prediction Endpoint

### Endpoint

```text
POST /v1/predict
```

### Content Type

```text
application/json
```

---

## Request Example

```bash
curl -X POST https://superkart-sales-forecasting.onrender.com/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Product_Weight": 9.3,
    "Product_Sugar_Content": "Low Sugar",
    "Product_Allocated_Area": 0.016,
    "Product_MRP": 249.8,
    "Store_Size": "Medium",
    "Store_Location_City_Type": "Tier 2",
    "Store_Type": "Supermarket Type1",
    "Store_Age_Years": 10,
    "Product_Category": "Perishable",
    "Product_Family": "FD"
  }'
```

Example response:

```json
{
  "predicted_sales": 3646.94,
  "status": "success"
}
```

---

# 📥 API Input Parameters

| Feature                  | Type    | Example           |
| ------------------------ | ------- | ----------------- |
| Product_Weight           | Float   | 9.3               |
| Product_Sugar_Content    | String  | Low Sugar         |
| Product_Allocated_Area   | Float   | 0.016             |
| Product_MRP              | Float   | 249.8             |
| Store_Size               | String  | Medium            |
| Store_Location_City_Type | String  | Tier 2            |
| Store_Type               | String  | Supermarket Type1 |
| Store_Age_Years          | Integer | 10                |
| Product_Category         | String  | Perishable        |
| Product_Family           | String  | FD                |

---

# 🖥️ Streamlit Frontend

The frontend is built using Streamlit.

It provides an interactive interface for entering product and store information.

## Product Information

Users can enter:

* Product Weight
* Product Sugar Content
* Product Allocated Area
* Product MRP
* Product Category
* Product Family

## Store Information

Users can enter:

* Store Size
* Store Location City Type
* Store Type
* Store Age

After submitting the form, the frontend sends the data to the Flask backend and displays the predicted sales value.

---

# 🌐 Live Frontend

The production Streamlit application is available at:

**[https://superkart-sales-forecasting-1.onrender.com](https://superkart-sales-forecasting-1.onrender.com)**

Open the URL in a browser to use the application.

---

# 🔗 Frontend and Backend Communication

The frontend uses an environment variable to determine the backend URL.

The application contains:

```python
BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:7860"
).rstrip("/")

PREDICTION_URL = f"{BACKEND_URL}/v1/predict"
```

## Local Development

```text
BACKEND_URL=http://localhost:7860
```

## Production

```text
BACKEND_URL=https://superkart-sales-forecasting.onrender.com
```

This allows the same frontend code to work in both local and production environments.

---

# 🐳 Docker

Both backend and frontend applications are containerized using Docker.

---

## Backend Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY sales_prediction_api.py .
COPY superkart_sales_pipeline.joblib .

EXPOSE 10000

CMD ["sh", "-c", "gunicorn --workers 2 --bind 0.0.0.0:${PORT:-10000} sales_prediction_api:app"]
```

---

## Frontend Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 10000

CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-10000} --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false"]
```

---

# 💻 Local Development

## Prerequisites

Make sure the following are installed:

* Python 3.11+
* Docker
* Git

---

# Backend

Navigate to the backend directory:

```bash
cd backend
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
python sales_prediction_api.py
```

The API will be available at:

```text
http://localhost:7860
```

Test it:

```bash
curl http://localhost:7860/health
```

---

# Frontend

Open another terminal:

```bash
cd frontend
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set the backend URL.

### macOS / Linux

```bash
export BACKEND_URL=http://localhost:7860
```

### Windows PowerShell

```powershell
$env:BACKEND_URL="http://localhost:7860"
```

Run Streamlit:

```bash
streamlit run app.py
```

The frontend will be available at:

```text
http://localhost:8501
```

---

# 🐳 Docker Deployment Locally

## Build Backend Image

From the project root:

```bash
docker build -t superkart-sales-backend ./backend
```

Run the backend:

```bash
docker run -d \
  --name superkart-backend \
  -p 7860:7860 \
  superkart-sales-backend
```

---

## Build Frontend Image

```bash
docker build -t superkart-sales-frontend ./frontend
```

Run the frontend:

```bash
docker run -d \
  --name superkart-frontend \
  -p 8501:8501 \
  -e BACKEND_URL=http://host.docker.internal:7860 \
  superkart-sales-frontend
```

The frontend will be available at:

```text
http://localhost:8501
```

---

# ☁️ Cloud Deployment

The application is deployed on **Render** using Docker containers.

Two separate services are used:

```text
Backend Service
      │
      └── backend/

Frontend Service
      │
      └── frontend/
```

---

# Backend Render Configuration

The backend service uses:

```text
Root Directory:
backend

Runtime:
Docker
```

Dockerfile:

```text
backend/Dockerfile
```

Production URL:

```text
https://superkart-sales-forecasting.onrender.com
```

---

# Frontend Render Configuration

The frontend service uses:

```text
Root Directory:
frontend

Runtime:
Docker
```

Dockerfile:

```text
frontend/Dockerfile
```

Production URL:

```text
https://superkart-sales-forecasting-1.onrender.com
```

---

# 🔐 Environment Variables

The frontend Render service requires:

```text
BACKEND_URL=https://superkart-sales-forecasting.onrender.com
```

The backend URL should not be hard-coded into the frontend application.

The application reads it using:

```python
os.getenv("BACKEND_URL")
```

---

# 🔄 Production Prediction Flow

```text
User
 │
 ▼
Streamlit Frontend
 │
 │ JSON Request
 ▼
Render Backend
 │
 ▼
Flask API
 │
 ▼
Input Validation
 │
 ▼
Pandas DataFrame
 │
 ▼
Scikit-learn Pipeline
 │
 ├── StandardScaler
 │
 └── OneHotEncoder
 │
 ▼
Random Forest Regressor
 │
 ▼
Predicted Sales
 │
 ▼
JSON Response
 │
 ▼
Streamlit Frontend
 │
 ▼
User
```

---

# 🧪 Example Prediction

### Input

```json
{
  "Product_Weight": 9.3,
  "Product_Sugar_Content": "Low Sugar",
  "Product_Allocated_Area": 0.016,
  "Product_MRP": 249.8,
  "Store_Size": "Medium",
  "Store_Location_City_Type": "Tier 2",
  "Store_Type": "Supermarket Type1",
  "Store_Age_Years": 10,
  "Product_Category": "Perishable",
  "Product_Family": "FD"
}
```

### Output

```json
{
  "predicted_sales": 3646.94,
  "status": "success"
}
```

---

# 🛡️ Error Handling

The backend API handles different types of errors.

## Invalid Request

HTTP:

```text
400 Bad Request
```

Used when required fields are missing or input values are invalid.

## Server Error

HTTP:

```text
500 Internal Server Error
```

Used when an unexpected error occurs during prediction.

The API also includes logging for:

* Model loading
* Incoming prediction requests
* Validation errors
* Prediction failures
* Successful predictions

---

# 📈 Model Evaluation

The final model was evaluated using multiple regression metrics.

The optimized Random Forest model achieved:

```text
R² Score: 0.9317
```

This indicates that the model explains approximately **93.17% of the variance** in the target sales values on the evaluation dataset.

The optimized Random Forest model was therefore selected as the final production model.

---

# 🧰 Technology Stack

### Programming

* Python 3.11

### Data Science

* Pandas
* NumPy
* Scikit-learn
* Joblib

### Machine Learning

* Random Forest Regressor
* XGBoost
* GridSearchCV
* StandardScaler
* OneHotEncoder

### Backend

* Flask
* Gunicorn
* REST API

### Frontend

* Streamlit
* Requests

### Deployment

* Docker
* Render
* GitHub

---

# 📦 Frontend Dependencies

```text
streamlit==1.39.0
requests==2.32.3
```

---

# 🎯 Project Objectives

The main objectives of this project were:

* Perform exploratory data analysis
* Clean and preprocess retail data
* Engineer meaningful product and store features
* Train multiple regression models
* Optimize model hyperparameters
* Evaluate model performance
* Select the best-performing model
* Serialize the complete ML pipeline
* Build a REST API
* Build an interactive frontend
* Containerize the application
* Deploy the application to the cloud

---

# 🚀 Future Improvements

Potential improvements include:

* Add batch CSV prediction
* Add prediction history
* Add user authentication
* Add database integration
* Add model monitoring
* Add automated model retraining
* Add model versioning
* Add API documentation using Swagger/OpenAPI
* Add CI/CD automation
* Add sales analytics dashboards
* Add data drift monitoring

---

# 👨‍💻 Author

**Sankari**



---

# 🔗 Project Links

| Resource             | Link                                                                                                                 |
| -------------------- | -------------------------------------------------------------------------------------------------------------------- |
| 🌐 Live Application  | [https://superkart-sales-forecasting-1.onrender.com](https://superkart-sales-forecasting-1.onrender.com)             |
| 🔌 Backend API       | [https://superkart-sales-forecasting.onrender.com](https://superkart-sales-forecasting.onrender.com)                 |
| ❤️ API Health Check  | [https://superkart-sales-forecasting.onrender.com/health](https://superkart-sales-forecasting.onrender.com/health)   |
| 💻 GitHub Repository | [https://github.com/sankarirk/SuperKart-Sales-Forecasting](https://github.com/sankarirk/SuperKart-Sales-Forecasting) |

