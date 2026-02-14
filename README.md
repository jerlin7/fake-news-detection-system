# AI-Based Fake News Detection System

## Project Overview

This project is an AI-based system that detects whether a news article is **REAL** or **FAKE** using Natural Language Processing (NLP) techniques.

The system analyzes the text of a news article and provides:

* Prediction result (REAL or FAKE)
* Confidence score
* Highlighted suspicious phrases
* Generated explanation for the prediction
* Model evaluation metrics

This project was developed as part of a machine test to demonstrate basic NLP, machine learning, and explainable AI concepts.

---

## Objective

The main objective of this project is to build a simple automated system that can:

* Take a news article as input
* Classify it as real or fake
* Identify important or suspicious words
* Provide an explanation for the prediction
* Evaluate model performance

---

## Dataset

The dataset used is the **Fake and Real News Dataset** from Kaggle.

It contains:

* Real news articles
* Fake news articles
* News text used for training and testing the model

---

## Technologies Used

* Python
* Google Colab
* Pandas & NumPy
* Scikit-learn
* NLP (Text preprocessing & TF-IDF)
* Logistic Regression
* LIME (Explainable AI)

---

## Project Workflow

1. Load and combine real and fake news datasets
2. Perform text preprocessing and cleaning
3. Convert text into numerical features using TF-IDF
4. Train a machine learning classification model
5. Evaluate model performance using accuracy, precision, recall, and F1-score
6. Explain predictions using LIME
7. Generate a credibility explanation for the result

---

## Model Output

The system provides:

* Prediction: REAL or FAKE
* Confidence score
* Highlighted important words
* Generated explanation
* Evaluation metrics

---

## How to Run the Project

1. Open the notebook in Google Colab or Jupyter Notebook.
2. Upload the dataset files (`True.csv` and `Fake.csv`).
3. Run all cells in order.
4. Enter a news article in the input section to test the model.

---

## Future Improvements

* Use transformer-based models like BERT
* Build a web interface using Streamlit
* Improve explanation generation using advanced AI models

---

## Author

Beginner Machine Learning Project
