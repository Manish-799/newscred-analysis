import React, { useState } from "react";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState("");
  const [confidence, setConfidence] = useState("");

  const handleCheck = async () => {
    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Backend response:", data); 
      
      
      setResult(data.prediction === "Real" ? " Real News" : " Fake News");
      setConfidence(`Confidence: ${(data.confidence * 100).toFixed(1)}%`);
      
    } catch (error) {
      console.error("Error:", error);
      setResult("❌ Error checking news");
      setConfidence("");
    }
  };

  return (
    <div style={{ padding: "50px", textAlign: "center", fontFamily: "sans-serif" }}>
      <h1>Fake News Detector</h1>
      <textarea
        rows="8"
        cols="60"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter news content here..."
      />
      <br />
      <button onClick={handleCheck} style={{ marginTop: "10px", padding: "10px 20px" }}>
        Check
      </button>
      <h2 style={{ marginTop: "20px" }}>{result}</h2>
      {confidence && <p style={{ fontSize: "18px", color: "#666" }}>{confidence}</p>}
    </div>
  );
}

export default App;