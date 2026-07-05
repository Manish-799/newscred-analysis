# News Credibility Analysis System

An end-to-end NLP application for classifying news articles based on linguistic patterns learned from labelled news data.

The system combines a **TF-IDF text representation**, **Logistic Regression classifier**, **FastAPI inference API**, and an interactive **React dashboard** with local model explanations.

> This project performs supervised text classification. It does not independently verify factual claims or perform real-time fact checking.

---

## Application Preview

### News Analysis Dashboard

![News analysis dashboard](screenshots/analyzer.png)

### Prediction Distribution

![Prediction result](screenshots/model_prediction.png)

### Local Model Explanation

![Model explanation](screenshots/model_analysis_information.png)

### Local History and Model Training Numbers
![Local History and Model Numbers](screenshots/model_info_and_local_history.png)

---

## Overview

The application accepts a news headline or full article and analyses its text using a trained NLP classification pipeline.

The system returns:

- A `LIKELY_REAL`, `LIKELY_FAKE`, or `UNCERTAIN` verdict
- Fake and Real class probabilities
- The strongest active features pushing the prediction toward each class
- Article statistics
- Model evaluation information
- Browser-local analysis history

The application also exposes the trained model through a documented FastAPI REST API.

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
      +--------------------+
      |                    |
      v                    v
Class Probabilities    Feature Contributions
      |                    |
      +----------+---------+
                 |
                 v
          FastAPI REST API
                 |
                 v
          React Dashboard
```

---

## Machine Learning Pipeline

### 1. Dataset Preparation

The training pipeline loads labelled Real and Fake news datasets.

For each article:

```text
content = title + article body
```

The pipeline then:

1. Combines article titles and bodies.
2. Normalizes text using shared preprocessing logic.
3. Removes blank articles.
4. Removes duplicate articles using normalized article content.
5. Randomly shuffles the dataset.
6. Creates a stratified 80/20 train-test split.

After duplicate removal, the dataset contains:

```text
Total usable articles: 38,826

Real: 20,926
Fake: 17,900
```

---

### 2. Text Preprocessing

Training and inference share the same preprocessing function.

The preprocessing pipeline removes:

- URLs
- HTML tags
- Reuters-style publication datelines
- News-agency markers
- Common image and media attribution artifacts
- Unsupported special characters
- Repeated whitespace

Using shared preprocessing prevents training and inference transformations from diverging.

---

### 3. TF-IDF Feature Extraction

The classifier uses `TfidfVectorizer` with:

```text
Vocabulary size:     8,000
N-gram range:        (1, 2)
Maximum document frequency: 0.8
Minimum document frequency: 5
Sublinear term frequency:   Enabled
English stop words:         Removed
```

Both **unigrams and bigrams** are used.

Example features may include:

```text
said
government
president donald
white house
foreign minister
```

---

### 4. Classification Model

The system uses a class-balanced **Logistic Regression** classifier.

```text
Model:          Logistic Regression
Solver:         LBFGS
Class weights:  Balanced
Max iterations: 2000
```

A linear classifier was selected because it works effectively with sparse TF-IDF representations and allows direct inspection of learned feature coefficients.

---

## Model Evaluation

The model is evaluated on a held-out stratified test split.

```text
Training rows: 31,060
Testing rows:   7,766
```

### Results

| Metric | Value |
|---|---:|
| Accuracy | 97.36% |
| ROC-AUC | 0.9955 |
| Fake Precision | 0.98 |
| Fake Recall | 0.97 |
| Fake F1 | 0.97 |
| Real Precision | 0.97 |
| Real Recall | 0.98 |
| Real F1 | 0.98 |

### Confusion Matrix

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

The API loads model metadata directly from this saved evaluation artifact.

---

## Explainable Predictions

Because Logistic Regression is a linear classifier, the contribution of an active feature to the decision score can be inspected directly.

For a feature `i`:

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
Negative contribution -> pushes prediction toward Fake
Positive contribution -> pushes prediction toward Real
```

The API ranks the strongest active contributions for each prediction.

Example:

```text
Fake indicators

watch       -0.1342
doing       -0.1341
obama       -0.0926


Real indicators

said        +0.2772
monday      +0.1674
president   +0.0687
```

These explanations are derived from the fitted model and the article's active TF-IDF features.

They are not hardcoded keyword rules.

---

## Uncertain Predictions

The underlying classifier is binary:

```text
Fake
Real
```

The API adds an application-level abstention rule.

If the highest model class probability is below `0.60`, the displayed verdict becomes:

```text
UNCERTAIN
```

For example:

```text
Fake probability: 0.53
Real probability: 0.47

Verdict: UNCERTAIN
```

The raw model prediction is still preserved in the API response.

The `0.60` uncertainty threshold is currently an application heuristic and is not presented as an optimized or calibrated statistical threshold.

---

## Backend API

The backend is built using **FastAPI**.

Start the API with:

```bash
python -m uvicorn backend.app:app --reload
```

The API runs at:

```text
http://127.0.0.1:8000
```

Interactive Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Artifact and API health information |
| GET | `/model-info` | Model and evaluation metadata |
| POST | `/predict` | Analyse news text |

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

## Frontend Features

The React dashboard provides:

- News headline and article input
- Live word and character counts
- Prediction verdict
- Fake and Real class probability visualization
- Local TF-IDF feature explanations
- Article statistics
- Model evaluation information
- Classification pipeline visualization
- Loading and API error states
- Responsive interface
- Recent analysis history using browser `localStorage`

Recent analysis history is stored only in the user's browser.

No database is used for frontend history.

---

## Project Structure

```text
fake-news-detection-system/
|
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
│   ├── src/
│   │   ├── App.js
│   │   ├── App.css
│   │   └── index.js
│   │
│   ├── package.json
│   └── package-lock.json
│
├── screenshots/
│   ├── analyzer.png
│   ├── prediction.png
│   └── explanation.png
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Running the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Manish-799/fake-news-detection-system-1.git
cd fake-news-detection-system-1
```

---

### 2. Create a Python Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

---

### 3. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Start the FastAPI Backend

```bash
python -m uvicorn backend.app:app --reload
```

The backend will run at:

```text
http://127.0.0.1:8000
```

---

### 5. Install Frontend Dependencies

Open another terminal:

```bash
cd frontend
npm install
```

---

### 6. Start the React Frontend

```bash
npm start
```

The React application will run at:

```text
http://localhost:3000
```

---

## Retraining the Model

The trained model artifacts are included in the repository, so retraining is not required to run the application.

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

The training script will generate:

```text
backend/artifacts/
├── model_final.pkl
├── vectorizer_final.pkl
└── metrics.json
```

See [`dataset/README.md`](dataset/README.md) for dataset requirements.

---

## Technology Stack

### Machine Learning

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression

### Backend

- FastAPI
- Pydantic
- Uvicorn

### Frontend

- React
- JavaScript
- CSS
- Browser localStorage

---

## Model Limitations

This classifier learns statistical linguistic patterns from labelled text.

It does **not**:

- Independently verify factual claims
- Search external sources
- Analyse publisher reputation
- Perform real-time fact checking
- Determine objective truth from evidence

Feature auditing also showed that the dataset contains differences in writing style, editorial tone, topic distribution, and publication patterns between classes.

Obvious structural artifacts such as Reuters datelines and media attribution markers were removed during preprocessing. However, source and domain correlations may still remain.

Because the training corpus is dominated by a specific historical and political news distribution, predictions may not generalize reliably to unseen publishers, geographic domains, or newer news cycles.

The reported `97.36%` accuracy represents performance on the held-out split of this labelled dataset and should not be interpreted as 97.36% real-world misinformation detection accuracy.

---

## Future Improvements

Potential improvements include:

- Probability calibration
- Validation-based abstention threshold selection
- Source-aware evaluation splits
- Cross-domain news evaluation
- Transformer-based text classification
- External evidence retrieval
- Source credibility analysis
- Temporal dataset updates

---

## Author

**Manish Arora**

Computer Science Engineering student interested in machine learning, backend engineering, and applied software systems.