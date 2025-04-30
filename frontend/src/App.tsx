// src/App.tsx
import React, { useState } from "react";
import axios from "axios";
import "./App.css"; //  custom CSS file

export default function App() {
  const [inputText, setInputText] = useState("");
  const [themes, setThemes] = useState<string[]>([]);
  const [advice, setAdvice] = useState("");
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setLoading(true);
    setAdvice("");
    try {
      const res = await axios.post("http://localhost:8005/predict",  {
        input_text: inputText,
      });
      setThemes(res.data.predicted_themes);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateAdvice = async () => {
    setLoading(true);
    try {
      const res = await axios.post("http://localhost:8005/generate-advice", {
        input_text: inputText,
        predicted_themes: themes,
      });
      setAdvice(res.data.advice);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header className="header">
        <h1>ChatCounselorAI</h1>
        <p>
          ChatCounselorAI is a ML-powered LLM that was adapted from mental health counseling
          transcripts to help support mental health counselors to provide helpful advice to
          their patients based on thematic issues.
        </p>
      </header>

      <div className="main">
        <textarea
          rows={5}
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          className="textarea"
          placeholder="Enter your notes about the patient here..."
        />

        <div className="button-group">
          <button onClick={handlePredict} className="predict-button">
            Predict Themes
          </button>

          <button
            onClick={handleGenerateAdvice}
            className="advice-button"
            disabled={themes.length === 0}
          >
            Generate Advice
          </button>
        </div>

        {loading && <p className="loading">Loading...</p>}

        {themes.length > 0 && (
          <div className="output-box">
            <h2>Predicted Themes:</h2>
            <ul>
              {themes.map((theme) => (
                <li key={theme}>{theme}</li>
              ))}
            </ul>
          </div>
        )}

        {advice && (
          <div className="output-box">
            <h2>LLM Advice:</h2>
            <p>{advice}</p>
          </div>
        )}
      </div>
    </div>
  );
}