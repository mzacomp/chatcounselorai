import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
import joblib
import os

def main():
    print("Multi-label training script is running...")


# Get the absolute path to the CSV
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw/labeled_theme_data_expanded.csv"))

print(" Looking for file at:", csv_path)

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File not found at: {csv_path}")

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

print(f"CSV loaded with shape: {df.shape}")
print(" Sample rows:")
print(df.head())

# Preprocess multi-label themes
df["Theme"] = df["Theme"].apply(lambda x: [t.strip() for t in x.split(",")])

 # Features and labels
X = df["Context"]
y_raw = df["Theme"]

# Convert labels to multi-hot encoding
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y_raw)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=10000,ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# One-vs-Rest classifier for multilabel setup
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train_vec, y_train)

 # Predict and evaluate
y_pred = model.predict(X_test_vec)
print("\n Classification Report (multi-label):")
print(classification_report(y_test, y_pred, target_names=mlb.classes_))

# Save model artifacts
os.makedirs("trained_model", exist_ok=True)
joblib.dump(model, "trained_model/theme_multilabel_classifier.joblib")
joblib.dump(vectorizer, "trained_model/tfidf_vectorizer.joblib")
joblib.dump(mlb, "trained_model/multilabel_binarizer.joblib")

print("\n All artifacts saved to 'trained_model/'")
print(" Multi-label model training complete")

if __name__ == "__main__":
    main()
