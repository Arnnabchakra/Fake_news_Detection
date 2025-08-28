# Fake_news_Detection

This repository contains code for a machine learning project focused on detecting fake news. The project utilizes Python and various libraries such as Pandas, NumPy, Matplotlib, NLTK, and Scikit-learn.

**Kaggle :** https://www.kaggle.com/code/alihassanml/fake-news-detection


## 📌 Overview  
Fake news is a major problem in today's digital world, where misinformation spreads rapidly through social media and online platforms. This project focuses on building a **Machine Learning model** to classify news articles as **Real** or **Fake** based on their textual content.

The project uses **Natural Language Processing (NLP)** techniques combined with **Logistic Regression** for classification.

📌 **Kaggle Notebook:** [Click Here](https://www.kaggle.com/code/alihassanml/fake-news-detection)  

---

## ⚙️ Tech Stack & Libraries  
- **Python** (Core Programming Language)  
- **pandas** – Data manipulation & analysis  
- **numpy** – Numerical computations  
- **matplotlib** & **seaborn** – Data visualization  
- **re** – Regular Expressions for text cleaning  
- **NLTK** – Natural Language Processing (Stopwords, Stemming)  
- **scikit-learn** – Machine Learning (TF-IDF, Logistic Regression, Accuracy Score)  


## Dataset

The dataset used in this project contains a collection of news articles labeled as either fake or real. It includes various features such as the title, text, and other metadata.

## Approach

1. **Data Preprocessing**: Text data is cleaned and preprocessed using techniques such as removing stopwords, stemming, and vectorization.
2. **Feature Engineering**: Text features are extracted using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.
3. **Model Training**: The preprocessed data is split into training and testing sets. A logistic regression model is trained on the training data.
4. **Model Evaluation**: The trained model is evaluated on the test set using accuracy as the performance metric.


## 🛠 Project Workflow  

### ✅ 1. Data Preprocessing  
✔ Remove unwanted characters, symbols, URLs  
✔ Tokenization  
✔ Stopword Removal  
✔ Stemming  

### ✅ 2. Feature Extraction  
✔ Convert text into numerical vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)**  

### ✅ 3. Model Training  
✔ Split data into **training** and **testing sets**  
✔ Train a **Logistic Regression** model  

### ✅ 4. Model Evaluation  
✔ Evaluate using **Accuracy Score**  

### ✅ 5. Deployment  
✔ Use trained model to predict new data  

---

## 📊 Accuracy  
The **Logistic Regression** model achieves **high accuracy** in detecting fake news based on textual content.  



## Repository Structure

- `data/`: Contains the dataset used in the project.
- `notebooks/`: Jupyter notebooks containing code for data preprocessing, model training, and evaluation.
- `scripts/`: Python scripts for various functions and utilities used in the project.
- `README.md`: This file, providing an overview of the project.


Feel free to contribute to this project by opening issues or pull requests.
