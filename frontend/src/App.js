import React, { useEffect, useState } from "react";

import "./App.css";


const API_BASE_URL =
  process.env.REACT_APP_API_BASE_URL
  || "http://127.0.0.1:8000";
const HISTORY_KEY = "news-analysis-history";
const MAX_HISTORY_ITEMS = 6;


function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);

  const [loading, setLoading] = useState(false);
  const [modelInfoLoading, setModelInfoLoading] = useState(true);

  const [error, setError] = useState("");
  const [recentAnalyses, setRecentAnalyses] = useState([]);


  // -------------------------------------------------------
  // INITIAL DATA
  // -------------------------------------------------------

  useEffect(() => {
    loadModelInfo();
    loadHistory();
  }, []);


  // -------------------------------------------------------
  // MODEL INFORMATION
  // -------------------------------------------------------

  const loadModelInfo = async () => {
    setModelInfoLoading(true);

    try {
      const response = await fetch(
        `${API_BASE_URL}/model-info`
      );

      if (!response.ok) {
        throw new Error(
          "Unable to load model information."
        );
      }

      const data = await response.json();

      setModelInfo(data);
    } catch (requestError) {
      console.error(
        "Model information error:",
        requestError
      );
    } finally {
      setModelInfoLoading(false);
    }
  };


  // -------------------------------------------------------
  // LOCAL HISTORY
  // -------------------------------------------------------

  const loadHistory = () => {
    try {
      const savedHistory = localStorage.getItem(
        HISTORY_KEY
      );

      if (!savedHistory) {
        return;
      }

      const parsedHistory = JSON.parse(
        savedHistory
      );

      if (Array.isArray(parsedHistory)) {
        setRecentAnalyses(parsedHistory);
      }
    } catch (storageError) {
      console.error(
        "Unable to load analysis history:",
        storageError
      );

      localStorage.removeItem(HISTORY_KEY);
    }
  };


  const saveAnalysis = (
    articleText,
    analysisResult
  ) => {
    const historyItem = {
      id: Date.now(),
      text: articleText,
      verdict: analysisResult.verdict,
      probability: analysisResult.model_probability,
      createdAt: new Date().toISOString(),
    };

    setRecentAnalyses((currentHistory) => {
      const updatedHistory = [
        historyItem,
        ...currentHistory,
      ].slice(0, MAX_HISTORY_ITEMS);

      localStorage.setItem(
        HISTORY_KEY,
        JSON.stringify(updatedHistory)
      );

      return updatedHistory;
    });
  };


  const clearHistory = () => {
    localStorage.removeItem(HISTORY_KEY);
    setRecentAnalyses([]);
  };


  // -------------------------------------------------------
  // ARTICLE STATISTICS
  // -------------------------------------------------------

  const trimmedText = text.trim();

  const wordCount = trimmedText
    ? trimmedText.split(/\s+/).length
    : 0;

  const characterCount = text.length;


  // -------------------------------------------------------
  // ANALYSIS
  // -------------------------------------------------------

  const handleAnalyze = async () => {
    if (!trimmedText) {
      setError(
        "Enter a headline or news article before analysing."
      );

      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/predict`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            text: trimmedText,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          extractErrorMessage(data)
        );
      }

      setResult(data);

      saveAnalysis(
        trimmedText,
        data
      );
    } catch (requestError) {
      console.error(
        "Analysis request failed:",
        requestError
      );

      setError(
        requestError.message
        || "Unable to analyse the article."
      );
    } finally {
      setLoading(false);
    }
  };


  const extractErrorMessage = (data) => {
    if (typeof data?.detail === "string") {
      return data.detail;
    }

    if (Array.isArray(data?.detail)) {
      return data.detail
        .map((item) => item.msg)
        .join(" ");
    }

    return "Unable to analyse the article.";
  };


  const clearAnalysis = () => {
    setText("");
    setResult(null);
    setError("");
  };


  const openHistoryItem = (item) => {
    setText(item.text);
    setResult(null);
    setError("");

    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  };


  // -------------------------------------------------------
  // DISPLAY HELPERS
  // -------------------------------------------------------

  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(1)}%`;
  };


  const formatVerdict = (verdict) => {
    return verdict
      .replaceAll("_", " ")
      .toLowerCase()
      .replace(/\b\w/g, (character) =>
        character.toUpperCase()
      );
  };


  const getVerdictClass = (verdict) => {
    if (verdict === "LIKELY_REAL") {
      return "verdict-real";
    }

    if (verdict === "LIKELY_FAKE") {
      return "verdict-fake";
    }

    return "verdict-uncertain";
  };


  const getHistoryExcerpt = (articleText) => {
    if (articleText.length <= 105) {
      return articleText;
    }

    return `${articleText.slice(0, 105)}...`;
  };


  // -------------------------------------------------------
  // RENDER
  // -------------------------------------------------------

  return (
    <div className="app-shell">
      <header className="hero">
        <div className="hero-content">
          <div className="brand-row">
            <div className="brand-mark">
              NC
            </div>

            <span className="brand-name">
              NewsCred
            </span>
          </div>

          <div className="hero-copy">
            <span className="eyebrow">
              NLP Classification System
            </span>

            <h1>
              News Credibility
              <span> Analysis</span>
            </h1>

            <p>
              Analyse linguistic patterns in news
              articles using TF-IDF and an explainable
              Logistic Regression classifier.
            </p>
          </div>

          <div className="hero-model-status">
            <span className="status-dot" />

            {modelInfoLoading
              ? "Loading model"
              : modelInfo
              ? "Model online"
              : "Model information unavailable"}
          </div>
        </div>
      </header>


      <main className="dashboard">
        <section className="analysis-card panel">
          <div className="panel-heading">
            <div>
              <span className="section-label">
                Article Analysis
              </span>

              <h2>
                Analyse news content
              </h2>

              <p>
                Paste a news headline or full article.
                Longer article text generally provides
                more linguistic features for the model.
              </p>
            </div>
          </div>


          <div className="textarea-wrapper">
            <textarea
              value={text}
              onChange={(event) =>
                setText(event.target.value)
              }
              placeholder="Paste a news headline or article here..."
              maxLength={100000}
            />

            <div className="input-stats">
              <span>
                {wordCount.toLocaleString()} words
              </span>

              <span>
                {characterCount.toLocaleString()} characters
              </span>
            </div>
          </div>


          {error && (
            <div className="error-message">
              <strong>
                Analysis unavailable
              </strong>

              <span>
                {error}
              </span>
            </div>
          )}


          <div className="action-row">
            <button
              className="secondary-button"
              type="button"
              onClick={clearAnalysis}
              disabled={loading}
            >
              Clear
            </button>

            <button
              className="primary-button"
              type="button"
              onClick={handleAnalyze}
              disabled={loading || !trimmedText}
            >
              {loading
                ? "Analysing article..."
                : "Analyse Article"}
            </button>
          </div>
        </section>


        {result && (
          <section className="results-section">
            <div className="result-grid">
              <article
                className={`
                  verdict-card
                  panel
                  ${getVerdictClass(result.verdict)}
                `}
              >
                <span className="section-label">
                  Analysis Result
                </span>

                <div className="verdict-content">
                  <div className="verdict-badge">
                    {formatVerdict(
                      result.verdict
                    )}
                  </div>

                  <div className="probability-number">
                    {formatPercentage(
                      result.model_probability
                    )}
                  </div>

                  <p>
                    Model probability for the predicted
                    class
                  </p>
                </div>

                <div className="raw-prediction">
                  Raw model prediction

                  <strong>
                    {result.model_prediction}
                  </strong>
                </div>
              </article>


              <article className="probability-card panel">
                <span className="section-label">
                  Class Probabilities
                </span>

                <h2>
                  Prediction distribution
                </h2>

                <div className="probability-list">
                  <ProbabilityBar
                    label="Fake class"
                    value={
                      result.probabilities.fake
                    }
                    type="fake"
                  />

                  <ProbabilityBar
                    label="Real class"
                    value={
                      result.probabilities.real
                    }
                    type="real"
                  />
                </div>

                <p className="probability-note">
                  Probabilities represent model output,
                  not independently verified factual truth.
                </p>
              </article>
            </div>


            <article className="explanation-card panel">
              <div className="panel-heading">
                <div>
                  <span className="section-label">
                    Model Explanation
                  </span>

                  <h2>
                    Features influencing this prediction
                  </h2>

                  <p>
                    Active TF-IDF features are ranked by
                    their signed contribution to the
                    Logistic Regression decision score.
                  </p>
                </div>
              </div>


              <div className="explanation-grid">
                <FeatureColumn
                  title="Fake indicators"
                  subtitle="Features pushing toward Fake"
                  items={
                    result.explanation.fake_indicators
                  }
                  type="fake"
                />

                <FeatureColumn
                  title="Real indicators"
                  subtitle="Features pushing toward Real"
                  items={
                    result.explanation.real_indicators
                  }
                  type="real"
                />
              </div>
            </article>


            <article className="stats-card panel">
              <span className="section-label">
                Article Statistics
              </span>

              <div className="stats-grid">
                <StatItem
                  label="Words"
                  value={
                    result.article_stats.word_count
                  }
                />

                <StatItem
                  label="Characters"
                  value={
                    result.article_stats.character_count
                  }
                />

                <StatItem
                  label="Active model features"
                  value={
                    result.article_stats
                      .analyzed_feature_count
                  }
                />
              </div>
            </article>
          </section>
        )}


        <section className="model-card panel">
          <div className="panel-heading model-heading">
            <div>
              <span className="section-label">
                Model Information
              </span>

              <h2>
                Classification pipeline
              </h2>

              <p>
                Evaluation metrics are loaded directly
                from the saved training artifact.
              </p>
            </div>

            <div className="model-pill">
              API v2.0
            </div>
          </div>


          {modelInfo ? (
            <div className="model-grid">
              <ModelMetric
                label="Held-out accuracy"
                value={formatPercentage(
                  modelInfo.accuracy
                )}
              />

              <ModelMetric
                label="ROC-AUC"
                value={
                  modelInfo.roc_auc.toFixed(4)
                }
              />

              <ModelMetric
                label="Vocabulary"
                value={
                  modelInfo.vocabulary_size
                    .toLocaleString()
                }
              />

              <ModelMetric
                label="Training rows"
                value={
                  modelInfo.train_rows.toLocaleString()
                }
              />

              <ModelMetric
                label="Test rows"
                value={
                  modelInfo.test_rows.toLocaleString()
                }
              />

              <ModelMetric
                label="Model"
                value={modelInfo.model_type}
              />
            </div>
          ) : (
            <div className="model-unavailable">
              Model metadata is currently unavailable.
              The prediction API may still be running.
            </div>
          )}


          {modelInfo && (
            <div className="pipeline-row">
              <span>
                Article
              </span>

              <span className="pipeline-arrow">
                →
              </span>

              <span>
                Text preprocessing
              </span>

              <span className="pipeline-arrow">
                →
              </span>

              <span>
                TF-IDF
              </span>

              <span className="pipeline-arrow">
                →
              </span>

              <span>
                Logistic Regression
              </span>

              <span className="pipeline-arrow">
                →
              </span>

              <span>
                Explanation
              </span>
            </div>
          )}
        </section>


        <section className="history-card panel">
          <div className="panel-heading history-heading">
            <div>
              <span className="section-label">
                Recent Analysis
              </span>

              <h2>
                Local analysis history
              </h2>

              <p>
                Stored only in this browser using
                localStorage.
              </p>
            </div>

            {recentAnalyses.length > 0 && (
              <button
                type="button"
                className="text-button"
                onClick={clearHistory}
              >
                Clear history
              </button>
            )}
          </div>


          {recentAnalyses.length === 0 ? (
            <div className="empty-history">
              No analyses yet. Completed predictions
              will appear here.
            </div>
          ) : (
            <div className="history-list">
              {recentAnalyses.map((item) => (
                <button
                  className="history-item"
                  type="button"
                  key={item.id}
                  onClick={() =>
                    openHistoryItem(item)
                  }
                >
                  <div className="history-copy">
                    <strong>
                      {getHistoryExcerpt(item.text)}
                    </strong>

                    <span>
                      {new Date(
                        item.createdAt
                      ).toLocaleString()}
                    </span>
                  </div>

                  <div className="history-result">
                    <span
                      className={`
                        history-verdict
                        ${getVerdictClass(
                          item.verdict
                        )}
                      `}
                    >
                      {formatVerdict(
                        item.verdict
                      )}
                    </span>

                    <strong>
                      {formatPercentage(
                        item.probability
                      )}
                    </strong>
                  </div>
                </button>
              ))}
            </div>
          )}
        </section>


        <section className="limitation-note">
          <strong>
            Model limitation
          </strong>

          <p>
            This system performs supervised text
            classification using linguistic patterns
            learned from labelled data. It does not
            independently verify claims, inspect external
            sources, or perform real-time fact checking.
          </p>
        </section>
      </main>


      <footer className="footer">
        News Credibility Analysis System
        <span>
          TF-IDF · Logistic Regression · FastAPI · React
        </span>
      </footer>
    </div>
  );
}


function ProbabilityBar({
  label,
  value,
  type,
}) {
  const percentage = value * 100;

  return (
    <div className="probability-item">
      <div className="probability-header">
        <span>
          {label}
        </span>

        <strong>
          {percentage.toFixed(1)}%
        </strong>
      </div>

      <div className="probability-track">
        <div
          className={`
            probability-fill
            probability-${type}
          `}
          style={{
            width: `${percentage}%`,
          }}
        />
      </div>
    </div>
  );
}


function FeatureColumn({
  title,
  subtitle,
  items,
  type,
}) {
  return (
    <div className="feature-column">
      <div className="feature-header">
        <div
          className={`
            feature-direction
            feature-direction-${type}
          `}
        />

        <div>
          <h3>
            {title}
          </h3>

          <p>
            {subtitle}
          </p>
        </div>
      </div>


      {items.length === 0 ? (
        <div className="empty-features">
          No active features in this direction.
        </div>
      ) : (
        <div className="feature-list">
          {items.map((item) => (
            <div
              className="feature-item"
              key={`${type}-${item.term}`}
            >
              <div className="feature-term">
                <strong>
                  {item.term}
                </strong>

                <span>
                  coefficient{" "}
                  {item.coefficient.toFixed(3)}
                </span>
              </div>

              <div className="feature-contribution">
                <span>
                  contribution
                </span>

                <strong
                  className={`contribution-${type}`}
                >
                  {item.contribution > 0
                    ? "+"
                    : ""}
                  {item.contribution.toFixed(4)}
                </strong>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}


function StatItem({
  label,
  value,
}) {
  return (
    <div className="stat-item">
      <strong>
        {value.toLocaleString()}
      </strong>

      <span>
        {label}
      </span>
    </div>
  );
}


function ModelMetric({
  label,
  value,
}) {
  return (
    <div className="model-metric">
      <span>
        {label}
      </span>

      <strong>
        {value}
      </strong>
    </div>
  );
}


export default App;