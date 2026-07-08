# News Credibility Analysis System

[![CI](https://github.com/Manish-799/newscred-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/Manish-799/newscred-analysis/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.13.7-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688)
![React](https://img.shields.io/badge/React-Frontend-61DAFB)

An end-to-end **explainable NLP classification system** that analyses linguistic patterns in news articles using TF-IDF and Logistic Regression.

The project combines a trained machine learning pipeline, FastAPI inference API, local feature-contribution explanations, an interactive React dashboard, automated tests, and GitHub Actions CI.

**97.36% held-out accuracy · 0.9955 ROC-AUC · 38,826 usable articles**

> This project performs supervised text classification. It does not independently verify factual claims or perform real-time fact checking.

---

## Live Demo

**Application:** https://newscred-analysis.vercel.app

**API Documentation:** https://newscred-api.onrender.com/docs

> The free backend instance may require a short cold start after inactivity.

---

## Application Preview

### News Analysis Dashboard

![News analysis dashboard](screenshots/analyzer.png)

### Prediction Distribution

![Prediction result](screenshots/model_prediction.png)

### Local Model Explanation

![Model explanation](screenshots/model_analysis_information.png)

### Model Information and Local History

![Model information and local history](screenshots/model_info_and_local_history.png)

---

## Overview

The application accepts a news headline or full article and analyses the text using a trained NLP classification pipeline.

The system returns:

* A `LIKELY_REAL`, `LIKELY_FAKE`, or `UNCERTAIN` verdict
* Fake and Real class probabilities
* The raw binary model prediction
* Strongest active features pushing the prediction toward each class
* Article word, character, and active-feature statistics
* Saved model evaluation information
* Browser-local analysis history

The trained model is exposed through a documented FastAPI REST API and consumed by a React frontend.

---

## Key Features

### Machine Learning

* TF-IDF text representation
* Unigram and bigram features
* Class-balanced Logistic Regression
* Shared preprocessing between training and inference
* Duplicate article removal
* Stratified train-test evaluation
* Saved model, vectorizer, and evaluation artifacts

### Explainability

* Per-prediction TF-IDF feature contributions
* Separate Fake and Real direction indicators
* Direct use of fitted Logistic Regression coefficients
* No hardcoded explanation keywords

### Backend

* FastAPI REST API
* Pydantic request and response validation
* Model artifact validation during application startup
* Health and model-information endpoints
* Explicit invalid-input handling

### Frontend

* React dashboard
* Article and headline analysis
* Live word and character counts
* Probability visualisation
* Feature-contribution explanations
* Model evaluation information
* Local analysis history using `localStorage`
* Responsive interface
* Loading and API error states

### Engineering

* 15 automated pytest tests
* API behaviour tests
* Text preprocessing unit tests
* Explainability invariant tests
* GitHub Actions continuous integration
* Automated backend test execution
* Automated React production build validation
* Frontend split into reusable components and API services

---

## Architecture

```text
News Article
      |
      v
Text Preprocessing
      |
      v
TF-IDF Vectorization
(Unigrams + Bigrams)
      |
      v
Logistic Regression
      |
      +------------------------+
      |                        |
      v                        v
Class Probabilities     Feature Contributions
      |                        |
      +------------+-----------+
                   |
                   v
            FastAPI REST API
                   |
                   v
             React Dashboard
```

The frontend communicates with the inference API through a dedicated API service layer.

```text
React Components
       |
       v
services/api.js
       |
       v
FastAPI API
       |
       v
TF-IDF + Logistic Regression
```

---

# Machine Learning Pipeline

## 1. Dataset Preparation

The training pipeline loads labelled Real and Fake news datasets.

For each article:

```text
content = title + article body
```

The pipeline:

1. Combines article titles and bodies.
2. Normalizes text using shared preprocessing logic.
3. Removes blank articles.
4. Removes duplicate articles using normalized article content.
5. Randomly shuffles the dataset.
6. Creates a stratified 80/20 train-test split.

After duplicate removal:

```text
Total usable articles: 38,826

Real: 20,926
Fake: 17,900
```

---

## 2. Text Preprocessing

Training and inference use the same `clean_text` preprocessing function.

The preprocessing pipeline removes:

* URLs
* HTML tags
* Reuters-style publication datelines
* News-agency markers
* Common image and media attribution artifacts
* Unsupported special characters
* Repeated whitespace

Text is also normalized to lowercase.

Using shared preprocessing prevents training and inference transformations from silently diverging.

---

## 3. TF-IDF Feature Extraction

The classifier uses `TfidfVectorizer` with:

```text
Vocabulary size:             8,000
N-gram range:                (1, 2)
Maximum document frequency:  0.8
Minimum document frequency:  5
Sublinear term frequency:    Enabled
English stop words:          Removed
```

Both unigrams and bigrams are represented.

Example vocabulary features may include:

```text
said
government
president donald
white house
foreign minister
```

---

## 4. Classification Model

The system uses a class-balanced Logistic Regression classifier.

```text
Model:          Logistic Regression
Solver:         LBFGS
Class weights:  Balanced
Max iterations: 2000
C:              0.1
```

A linear classifier was selected because it works effectively with sparse TF-IDF representations and allows direct inspection of learned feature coefficients.

This enables the system to provide local feature-contribution explanations without introducing a separate explanation model.

---

# Model Evaluation

The model is evaluated on a held-out stratified test split.

```text
Training rows: 31,060
Testing rows:   7,766
```

## Results

| Metric         |  Value |
| -------------- | -----: |
| Accuracy       | 97.36% |
| ROC-AUC        | 0.9955 |
| Fake Precision |   0.98 |
| Fake Recall    |   0.97 |
| Fake F1        |   0.97 |
| Real Precision |   0.97 |
| Real Recall    |   0.98 |
| Real F1        |   0.98 |

## Confusion Matrix

```text
Rows = Actual
Columns = Predicted

              Fake    Real
Fake          3463     117
Real            88    4098
```

Evaluation metrics are saved to:

```text
backend/artifacts/metrics.json
```

The API loads model metadata directly from the saved evaluation artifact.

> The reported metrics represent performance on the held-out split of this labelled dataset. They should not be interpreted as equivalent real-world misinformation detection performance.

---

# Explainable Predictions

Because Logistic Regression is a linear classifier, the contribution of an active feature to the model's decision score can be inspected directly.

For feature `i`:

```text
feature contribution
=
TF-IDF value_i × model coefficient_i
```

The complete Logistic Regression decision score has the form:

```text
z = intercept + Σ(x_i × β_i)
```

For this model:

```text
0 = Fake
1 = Real
```

Therefore:

```text
Negative contribution → pushes the decision toward Fake
Positive contribution → pushes the decision toward Real
```

The API calculates contributions only for TF-IDF features active in the submitted article and ranks the strongest values in each direction.

Example:

```text
Fake indicators

watch       -0.1342
doing       -0.1341
obama       -0.0926
```

```text
Real indicators

said        +0.2772
monday      +0.1674
president   +0.0687
```

These explanations are derived from the fitted vectorizer, the article's active TF-IDF features, and the trained Logistic Regression coefficients.

They are **not hardcoded keyword rules**.

---

# Uncertain Predictions

The underlying machine learning classifier is binary:

```text
Fake
Real
```

The API adds an application-level abstention rule.

If the highest model class probability is below `0.60`, the displayed verdict becomes:

```text
UNCERTAIN
```

Example:

```text
Fake probability: 0.53
Real probability: 0.47

Verdict: UNCERTAIN
```

The raw model prediction is still included in the API response.

The `0.60` threshold is currently an application heuristic. It is **not presented as an optimized or statistically calibrated confidence threshold**.

---

# Backend API

The backend is built using FastAPI.

Start the API with:

```bash
python -m uvicorn backend.app:app --reload
```

The API runs locally at:

```text
http://127.0.0.1:8000
```

Interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## Endpoints

| Method | Endpoint      | Description                         |
| ------ | ------------- | ----------------------------------- |
| GET    | `/`           | API status                          |
| GET    | `/health`     | Artifact and API health information |
| GET    | `/model-info` | Model and evaluation metadata       |
| POST   | `/predict`    | Analyse news text                   |

---

## Prediction API

### Request

```json
{
  "text": "The article text to analyse..."
}
```

### Example Response

```json
{
  "verdict": "LIKELY_FAKE",
  "model_prediction": "FAKE",
  "predicted_label": 0,
  "model_probability": 0.8585,
  "probabilities": {
    "fake": 0.8585,
    "real": 0.1415
  },
  "explanation": {
    "fake_indicators": [
      {
        "term": "watch",
        "tfidf_value": 0.0516,
        "coefficient": -2.6029,
        "contribution": -0.1342
      }
    ],
    "real_indicators": [
      {
        "term": "said",
        "tfidf_value": 0.0453,
        "coefficient": 6.117,
        "contribution": 0.2772
      }
    ]
  },
  "article_stats": {
    "word_count": 356,
    "character_count": 2032,
    "analyzed_feature_count": 118
  }
}
```

---

# Frontend Architecture

The React frontend is divided into reusable presentation components and a dedicated API service.

## Components

### `AnalyzerForm.js`

Handles the article input interface, live input statistics, validation errors, and analysis controls.

### `PredictionResult.js`

Displays:

* Prediction verdict
* Predicted-class probability
* Fake and Real class distributions
* Local feature contributions
* Article statistics

### `ModelInformation.js`

Displays saved model metrics and the classification pipeline.

### `AnalysisHistory.js`

Displays browser-local prediction history and allows previous article text to be reopened.

## API Service

`services/api.js` owns frontend-backend communication.

It handles:

```text
GET /model-info
POST /predict
JSON serialization
HTTP response validation
FastAPI error extraction
```

This keeps networking concerns separate from the main React page orchestration.

`App.js` remains responsible for application-level state, history persistence, and connecting the frontend components.

---

# Testing

The project contains **15 automated pytest tests**.

```text
tests/
├── test_api.py
└── test_text_utils.py
```

## API Tests

The API suite verifies:

* Root endpoint behaviour
* Health endpoint behaviour
* Model metadata responses
* Blank article rejection
* Unknown TF-IDF feature rejection
* Valid prediction response structure
* Probability range validation
* Fake and Real probabilities summing to approximately `1.0`
* Valid verdict and prediction labels
* Signed feature-contribution directions

The explanation tests enforce an important model invariant:

```text
Fake indicator contribution < 0
Real indicator contribution > 0
```

## Preprocessing Tests

The text utility suite verifies:

* Lowercase normalization
* URL removal
* HTML tag removal
* Reuters-style dateline removal
* News-agency reference removal
* Whitespace normalization
* Unsupported-character removal

Run the test suite with:

```bash
python -m pytest
```

Current test suite:

```text
15 passed
```

Development and testing dependencies are defined in:

```text
requirements-dev.txt
```

---

# Continuous Integration

GitHub Actions automatically validates the project on pushes and pull requests.

The workflow is defined in:

```text
.github/workflows/ci.yml
```

The CI pipeline contains two independent jobs:

```text
                    GitHub Actions CI
                           |
              +------------+------------+
              |                         |
              v                         v
       Backend Tests              Frontend Build
              |                         |
      Python 3.13.7                 Node.js
      Install dependencies          npm ci
      python -m pytest              npm run build
```

## Backend Tests

The backend CI job:

1. Checks out the repository.
2. Configures Python 3.13.7.
3. Restores the pip dependency cache when available.
4. Installs development dependencies.
5. Runs the complete pytest suite.

## Frontend Build

The frontend CI job:

1. Checks out the repository.
2. Configures Node.js.
3. Restores the npm dependency cache when available.
4. Installs dependencies using `npm ci`.
5. Generates an optimized React production build.

A failed test or failed frontend build causes the CI workflow to fail.

---

# Browser-Local History

Recent analyses are stored using browser `localStorage`.

The application stores a limited set of recent analysis records containing:

```text
Article text
Displayed verdict
Predicted-class probability
Analysis timestamp
```

History exists only in the user's browser.

No server-side database or user account is required for analysis history.

---

# Project Structure

```text
newscred-analysis/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── backend/
│   ├── __init__.py
│   ├── app.py
│   ├── train_model.py
│   ├── text_utils.py
│   │
│   └── artifacts/
│       ├── model_final.pkl
│       ├── vectorizer_final.pkl
│       └── metrics.json
│
├── dataset/
│   └── README.md
│
├── frontend/
│   ├── public/
│   │
│   ├── src/
│   │   ├── components/
│   │   │   ├── AnalysisHistory.js
│   │   │   ├── AnalyzerForm.js
│   │   │   ├── ModelInformation.js
│   │   │   └── PredictionResult.js
│   │   │
│   │   ├── services/
│   │   │   └── api.js
│   │   │
│   │   ├── App.css
│   │   ├── App.js
│   │   └── index.js
│   │
│   ├── README.md
│   ├── package.json
│   └── package-lock.json
│
├── screenshots/
│   ├── analyzer.png
│   ├── model_analysis_information.png
│   ├── model_info_and_local_history.png
│   └── model_prediction.png
│
├── tests/
│   ├── test_api.py
│   └── test_text_utils.py
│
├── .gitignore
├── .python-version
├── README.md
├── requirements.txt
└── requirements-dev.txt
```

---

# Running the Project Locally

## 1. Clone the Repository

```bash
git clone https://github.com/Manish-799/newscred-analysis.git
cd newscred-analysis
```

---

## 2. Create a Python Virtual Environment

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

---

## 3. Install Backend Dependencies

To run the application:

```bash
pip install -r requirements.txt
```

For development and testing:

```bash
pip install -r requirements-dev.txt
```

---

## 4. Start the FastAPI Backend

```bash
python -m uvicorn backend.app:app --reload
```

The backend runs at:

```text
http://127.0.0.1:8000
```

Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 5. Install Frontend Dependencies

Open another terminal:

```bash
cd frontend
npm install
```

---

## 6. Start the React Frontend

```bash
npm start
```

The React application runs at:

```text
http://localhost:3000
```

---

# Running Tests

From the project root:

```bash
python -m pytest
```

Expected result:

```text
15 passed
```

---

# Production Frontend Build

From the `frontend` directory:

```bash
npm run build
```

This generates an optimized production build and is the same frontend build validation performed by CI.

---

# Retraining the Model

The trained model artifacts are included in the repository, so retraining is **not required** to run the application.

To retrain the model:

1. Download the Fake and Real News Dataset.
2. Place the files in the `dataset` directory:

```text
dataset/
├── True.csv
└── Fake.csv
```

3. Run:

```bash
python -m backend.train_model
```

The training script generates:

```text
backend/artifacts/
├── model_final.pkl
├── vectorizer_final.pkl
└── metrics.json
```

See `dataset/README.md` for dataset requirements.

---

# Technology Stack

## Machine Learning

* Python
* Pandas
* Scikit-learn
* TF-IDF
* Logistic Regression

## Backend

* FastAPI
* Pydantic
* Uvicorn

## Frontend

* React
* JavaScript
* CSS
* Browser `localStorage`

## Testing

* pytest
* FastAPI `TestClient`
* HTTPX

## CI

* GitHub Actions
* Automated pytest execution
* Automated React production builds
* pip and npm dependency caching

## Deployment

* Vercel
* Render

---

# Model Limitations

This classifier learns statistical linguistic patterns from labelled text.

It does **not**:

* Independently verify factual claims
* Search external sources
* Analyse publisher reputation
* Retrieve supporting or contradicting evidence
* Perform real-time fact checking
* Determine objective truth from evidence

Feature auditing showed that the dataset contains differences in:

* Writing style
* Editorial tone
* Topic distribution
* Publication patterns

Obvious structural artifacts such as Reuters datelines and common media attribution markers were removed during preprocessing.

However, source and domain correlations may still remain.

The training corpus is dominated by a specific historical and political news distribution. Predictions may therefore not generalize reliably to unseen publishers, geographic domains, emerging topics, or newer news cycles.

The reported `97.36%` accuracy represents performance on the held-out split of this labelled dataset.

It should **not** be interpreted as `97.36%` real-world misinformation detection accuracy.

---

# Future Improvements

Potential improvements include:

* Validation-based abstention threshold selection
* Probability calibration
* Source-aware evaluation splits
* Cross-domain news evaluation
* Temporal evaluation on newer articles
* Transformer-based classifier comparison
* External evidence retrieval
* Source credibility analysis
* Model and artifact version tracking

---

# Author

**Manish Arora**

Computer Science Engineering student interested in machine learning, backend engineering, and applied software systems.
