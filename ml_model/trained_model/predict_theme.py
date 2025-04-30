import joblib
import os
import numpy as np

def load_artifacts(model_dir="trained_model"):
    model = joblib.load(os.path.join(model_dir, "theme_multilabel_classifier.joblib"))
    vectorizer = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.joblib"))
    mlb = joblib.load(os.path.join(model_dir, "multilabel_binarizer.joblib"))
    return model, vectorizer, mlb

def predict_themes_top_k(text, model, vectorizer, mlb, top_k=3):
    # Preprocess and vectorize
    processed = [text.lower().strip()]
    vectorized = vectorizer.transform(processed)

    # Predict probabilities
    probs = model.predict_proba(vectorized)[0]

    # Get top K indices with non-zero probabilities
    top_indices = probs.argsort()[-top_k:][::-1]
    top_themes = [mlb.classes_[i] for i in top_indices if probs[i] > 0]

    return top_themes

def keyword_boost_filter(predicted_themes, input_text):
    # Simple keyword-to-theme mapping
    keywords = {
        "job": "Work Issue",
        "work": "Work Issue",
        "sleep": "Sleep Issues",
        "partner": "Marital/Relationship Problems",
        "relationship": "Marital/Relationship Problems",
        "worthless": "Self-Esteem Issues",
        "identity": "Existential / Identity",
        "lonely": "Social Isolation",
        "loss": "Grief or Loss",
        "anxious": "Anxiety",
        "anxiety": "Anxiety",
        "depressed": "Depression",
        "hopeless": "Depression",
        "stress": "Stress",
        "trauma": "Trauma",
        "school": "School Issue",
        "drugs": "Alcoholism/Drug Issue",
        "alcohol": "Alcoholism/Drug Issue"
    }

    boosted_themes = predicted_themes.copy()
    for word, theme in keywords.items():
        if word in input_text.lower() and theme not in boosted_themes:
            boosted_themes.append(theme)

    # Remove duplicates, keep order
    return list(dict.fromkeys(boosted_themes))

if __name__ == "__main__":
    model, vectorizer, mlb = load_artifacts()

    example_text = input("Enter a patient message:\n> ")

    # Top-K prediction
    themes = predict_themes_top_k(example_text, model, vectorizer, mlb, top_k=3)

    # Keyword-based boosting
    themes = keyword_boost_filter(themes, example_text)

    print("\n Final Predicted Themes (Logistic + Top-K + Keywords):")
    if themes:
        for t in themes:
            print(f" - {t}")
    else:
        print("No strong themes predicted.")
