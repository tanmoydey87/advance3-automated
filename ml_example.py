# File declares that parameter comes from uploaded file
# Form declares that parameter comes from HTML form upload
from fastapi import FastAPI, UploadFile, File, Form
import pandas as pd
from sklearn.linear_model import LinearRegression
import io

# app is an instance of FastAPI class

app = FastAPI()

# defines the GET endpoint at the root / URL and displays the message
@app.get("/")
def home():
    return {"message": "Welcome to Linear Regression API. Go to /docs to upload CSV"}

# defines the POST endpoint at the /train_model/ URL and trains the model. 
# i.e. http://localhost:8000/train_model/
# Here, user upload csv file, specify feature columns and target column.
# async def is used to handle asynchronous requests
# file is the uploaded file
# features is the comma-separated list of feature columns
# target is the target column name
@app.post("/train_model/")
async def train_model(
    file: UploadFile = File(...),
    features: str = Form(...),   # Comma-separated list of feature columns
    target: str = Form(...)      # Target column name
):
    """
    Upload a CSV, specify feature columns and target column.
    Example: features="col1,col2,col3" target="output"
    """
    try:
        # reads the uploaded file into the container's memory 
        # Read uploaded file into pandas DataFrame
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Parse user-specified features and target
        feature_cols = [col.strip() for col in features.split(",")]
        target_col = target.strip()

        # Validate column existence.[] around target_col is to convert it to list to add to features_cols
        for col in feature_cols + [target_col]:
            if col not in df.columns:
                return {"error": f"Column '{col}' not found in CSV."}

        X = df[feature_cols]
        y = df[target_col]

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        # Return model details and sample predictions
        predictions = model.predict(X.head(5))  # first 5 rows only
        return {
            "features": feature_cols,
            "target": target_col,
            "coefficients": model.coef_.tolist(),
            "intercept": float(model.intercept_),
            "sample_predictions": predictions.tolist()
        }

    except Exception as e:
        return {"error": str(e)}
