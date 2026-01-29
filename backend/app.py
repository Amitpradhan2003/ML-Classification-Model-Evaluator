from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import os

from data_clean import DataClean
from missing_imputation import MissingImputation
from encoding import Encoding
from train_test_split import DataSplit
from model_development import ModelDevelopment
from predictions import Predictions
from metrics import Metrics

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Absolute paths (IMPORTANT)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    model: str = Form(...)
):
    df = pd.read_csv(file.file)

    df = DataClean(df).clean_data()
    df = MissingImputation(df).impute()
    df = Encoding(df).encode()

    target = df.columns[-1]
    X_train, X_test, y_train, y_test = DataSplit(df, target).split()

    model_obj = ModelDevelopment(model)
    model_instance = model_obj.get_model()
    model_instance = model_obj.train(X_train, y_train)

    pred_obj = Predictions(model_instance)
    y_pred = pred_obj.predict(X_test)
    y_prob = pred_obj.predict_proba(X_test)

    metrics, roc_img, cm_img = Metrics().evaluate(y_test, y_pred, y_prob)

    return {
        "metrics": metrics,
        "roc_curve": roc_img,
        "confusion_matrix": cm_img
    }
