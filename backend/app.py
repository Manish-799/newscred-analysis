import json
import pickle
from pathlib import Path
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.text_utils import clean_text


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "model_final.pkl"
VECTORIZER_PATH = ARTIFACTS_DIR / "vectorizer_final.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

UNCERTAIN_THRESHOLD = 0.60
TOP_EXPLANATION_FEATURES = 5


# ---------------------------------------------------------
# ARTIFACT LOADING
# ---------------------------------------------------------

def load_pickle_artifact(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}"
        )

    with open(path, "rb") as file:
        return pickle.load(file)


def load_json_artifact(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


model = load_pickle_artifact(MODEL_PATH)

vectorizer = load_pickle_artifact(
    VECTORIZER_PATH
)

metrics = load_json_artifact(
    METRICS_PATH
)


# ---------------------------------------------------------
# ARTIFACT VALIDATION
# ---------------------------------------------------------

def validate_artifacts() -> None:
    model_classes = {
        int(label)
        for label in model.classes_
    }

    if model_classes != {0, 1}:
        raise RuntimeError(
            "Expected model classes {0, 1}, "
            f"received {model_classes}."
        )

    if model.coef_.shape[0] != 1:
        raise RuntimeError(
            "Expected a binary Logistic Regression "
            "model with one coefficient row."
        )

    feature_names = (
        vectorizer.get_feature_names_out()
    )

    model_feature_count = model.coef_.shape[1]
    vectorizer_feature_count = len(feature_names)

    if (
        model_feature_count
        != vectorizer_feature_count
    ):
        raise RuntimeError(
            "Model and vectorizer feature counts "
            "do not match."
        )


validate_artifacts()


FEATURE_NAMES = (
    vectorizer.get_feature_names_out()
)

MODEL_COEFFICIENTS = model.coef_[0]

CLASS_TO_INDEX = {
    int(label): index
    for index, label in enumerate(model.classes_)
}


# ---------------------------------------------------------
# FASTAPI APPLICATION
# ---------------------------------------------------------

app = FastAPI(
    title="News Credibility Analysis API",
    description=(
        "NLP classification API using TF-IDF "
        "and Logistic Regression to analyse "
        "linguistic patterns in news text."
    ),
    version="2.0.0",
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN")

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

if FRONTEND_ORIGIN:
    ALLOWED_ORIGINS.append(
        FRONTEND_ORIGIN.rstrip("/")
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------
# REQUEST MODELS
# ---------------------------------------------------------

class PredictionRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=100_000,
    )


# ---------------------------------------------------------
# RESPONSE MODELS
# ---------------------------------------------------------

class ProbabilityResponse(BaseModel):
    fake: float
    real: float


class FeatureContribution(BaseModel):
    term: str
    tfidf_value: float
    coefficient: float
    contribution: float


class ExplanationResponse(BaseModel):
    fake_indicators: list[FeatureContribution]
    real_indicators: list[FeatureContribution]


class ArticleStats(BaseModel):
    word_count: int
    character_count: int
    analyzed_feature_count: int


class PredictionResponse(BaseModel):
    verdict: str
    model_prediction: str
    predicted_label: int
    model_probability: float
    probabilities: ProbabilityResponse
    explanation: ExplanationResponse
    article_stats: ArticleStats


class ModelInfoResponse(BaseModel):
    model_type: str
    feature_extraction: str
    accuracy: float
    roc_auc: float
    train_rows: int
    test_rows: int
    vocabulary_size: int
    uncertain_threshold: float


# ---------------------------------------------------------
# EXPLANATION LOGIC
# ---------------------------------------------------------

def get_local_explanation(
    features,
    top_n: int = TOP_EXPLANATION_FEATURES,
) -> tuple[
    list[FeatureContribution],
    list[FeatureContribution],
]:
    row = features.getrow(0)

    fake_contributions = []
    real_contributions = []

    for feature_index, tfidf_value in zip(
        row.indices,
        row.data,
    ):
        coefficient = MODEL_COEFFICIENTS[
            feature_index
        ]

        contribution = (
            tfidf_value * coefficient
        )

        feature = FeatureContribution(
            term=str(
                FEATURE_NAMES[feature_index]
            ),
            tfidf_value=float(tfidf_value),
            coefficient=float(coefficient),
            contribution=float(contribution),
        )

        if contribution < 0:
            fake_contributions.append(feature)

        elif contribution > 0:
            real_contributions.append(feature)

    fake_contributions.sort(
        key=lambda feature: feature.contribution
    )

    real_contributions.sort(
        key=lambda feature: feature.contribution,
        reverse=True,
    )

    return (
        fake_contributions[:top_n],
        real_contributions[:top_n],
    )


# ---------------------------------------------------------
# VERDICT LOGIC
# ---------------------------------------------------------

def get_verdict(
    predicted_label: int,
    model_probability: float,
) -> str:
    if model_probability < UNCERTAIN_THRESHOLD:
        return "UNCERTAIN"

    if predicted_label == 1:
        return "LIKELY_REAL"

    return "LIKELY_FAKE"


# ---------------------------------------------------------
# API ROUTES
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "News Credibility Analysis API",
        "status": "running",
        "documentation": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "vectorizer_loaded": True,
        "metrics_loaded": True,
        "api_version": "2.0.0",
    }


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
)
def model_info():
    return ModelInfoResponse(
        model_type="Logistic Regression",
        feature_extraction=(
            "TF-IDF with unigrams and bigrams"
        ),
        accuracy=float(
            metrics["accuracy"]
        ),
        roc_auc=float(
            metrics["roc_auc"]
        ),
        train_rows=int(
            metrics["train_rows"]
        ),
        test_rows=int(
            metrics["test_rows"]
        ),
        vocabulary_size=int(
            metrics["vocabulary_size"]
        ),
        uncertain_threshold=(
            UNCERTAIN_THRESHOLD
        ),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
):
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=422,
            detail="Article text cannot be blank.",
        )

    features = vectorizer.transform([text])

    if features.nnz == 0:
        raise HTTPException(
            status_code=422,
            detail=(
                "The supplied text does not contain "
                "features recognised by the model."
            ),
        )

    predicted_label = int(
        model.predict(features)[0]
    )

    probabilities = model.predict_proba(
        features
    )[0]

    fake_probability = float(
        probabilities[
            CLASS_TO_INDEX[0]
        ]
    )

    real_probability = float(
        probabilities[
            CLASS_TO_INDEX[1]
        ]
    )

    model_probability = max(
        fake_probability,
        real_probability,
    )

    model_prediction = (
        "REAL"
        if predicted_label == 1
        else "FAKE"
    )

    verdict = get_verdict(
        predicted_label=predicted_label,
        model_probability=model_probability,
    )

    (
        fake_indicators,
        real_indicators,
    ) = get_local_explanation(
        features=features,
    )

    article_stats = ArticleStats(
        word_count=len(text.split()),
        character_count=len(text),
        analyzed_feature_count=int(
            features.nnz
        ),
    )

    return PredictionResponse(
        verdict=verdict,
        model_prediction=model_prediction,
        predicted_label=predicted_label,
        model_probability=model_probability,
        probabilities=ProbabilityResponse(
            fake=fake_probability,
            real=real_probability,
        ),
        explanation=ExplanationResponse(
            fake_indicators=fake_indicators,
            real_indicators=real_indicators,
        ),
        article_stats=article_stats,
    )