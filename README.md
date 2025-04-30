# ChatCounselorAI 

#  ChatCounselorAI: Mental Health Support Assistant



<img width="1285" alt="Screenshot 2025-04-30 at 3 15 15 PM" src="https://github.com/user-attachments/assets/4e1700d7-2e7c-40a9-8f9a-edbe4c1771a9" />






**ChatCounselorAI** is a full-stack AI-powered web app designed to help mental health counselors better support patients. It integrates a machine learning model for condition risk prediction and thematic topics and a large language model (LLM) to generate personalized counseling guidance.

---

## Features

-  Predicts mental health issues and thematic paradigms based on patient textual data (e.g., depression, anxiety, sleep)
-  Counselor-facing AI assistant using OpenAI LLM,GPT 3.5-Turbo for personalized suggestions
-  ML backend served via FastAPI
-  Frontend built with React and TypeScript
-  Cross-origin enabled API (CORS)
-  Secure `.env` usage for API keys

---
## Dataset 

Dataset was provided by Kaggle's NLP Mental Health Conversations dataset(link:https://www.kaggle.com/datasets/thedevastator/nlp-mental-health-conversations/data) 

## Data Analysis 

Thematic paradigms(categorical predictions) were predicted using TF-IDF vectorizer and Logistic Regression modeling. 

##  Tech Stack

| Layer       | Technology         |
|-------------|--------------------|
| Frontend    | React, TypeScript, CSS |
| Backend     | FastAPI, Python, Node.js    |
| ML Model    | Scikit-learn, Joblib |
| LLM API     | OpenAI GPT 3.5-Turbo (via `openai` Python SDK) |
| Deployment  | (Localhost) |

## Web Application

Option #2 and #3 of the Web Application choices were selected. Mental health counselors can use the ML model to predict thematic paradigms and these thematic paradigms inform the LLM Generated Advice to provide prescient advice based on these specific themes. 

## Video Explanation 


https://github.com/user-attachments/assets/1765a727-544c-4be6-9aac-3d2bf869b4d1


by Mahsan Zare




---



