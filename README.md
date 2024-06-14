# Fake News Prediction Model

## Overview

This project aims to detect fake news articles using machine learning techniques like logistic regression, decision tree and confusion matrix. By leveraging a dataset containing news article titles, texts, publication dates, and classification labels (0 for fake and 1 for real news), the model predicts the authenticity of news articles with high accuracy.

## Dataset

The dataset used for training and testing the model consists of the following features:
- **Title**: The title of the news article.
- **Text**: The main content of the news article.
- **Subject**: Category of the news article.
- **Date**: The publication date of the news article.
- **Class**: The label indicating whether the news is fake (0) or real (1).

## Project Structure

- `Code.ipynb`: The main Jupyter Notebook containing the code for the fake news detection model.

## Features

- **Data Preprocessing**: Cleaning and preparing the dataset for training the model.
- **Feature Extraction**: Converting text data into numerical features using techniques such as TF-IDF Vectorization.
- **Model Training**: Training a machine learning model to classify news articles as fake or real.
- **Model Evaluation**: Evaluating the performance of the model using metrics such as accuracy, precision, recall, and F1-score.

## Installation

To run the code, you need to have the following libraries installed:
- `numpy`
- `pandas`
- `scikit-learn`
- `nltk`
- `matplotlib`
- `seaborn`

You can install these libraries using pip:
```bash
pip install numpy pandas scikit-learn nltk matplotlib seaborn
```

## Usage

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Open the `Code.ipynb` file in Jupyter Notebook.
4. Run the notebook cells to execute the code.

## Results

The model's performance is evaluated using the following metrics:
- **Accuracy**: The percentage of correctly classified news articles.
- **Precision**: The percentage of true positive predictions among all positive predictions.
- **Recall**: The percentage of true positive predictions among all actual positive cases.
- **F1-Score**: The harmonic mean of precision and recall.

## Conclusion

The fake news detection model provides an effective solution for identifying fake news articles. By utilizing machine learning techniques, the model achieves high accuracy in predicting the authenticity of news, contributing to combating the spread of misinformation.

## Contact

For any inquiries or contributions, please contact Aarohi Sharma at aarohi2316@gmail.com.
