# Spaceship Titanic - Kaggle Competition

![Spaceship Titanic](https://storage.googleapis.com/kaggle-competitions/kaggle/36238/logos/header.png?t=2022-06-23-19-57-45)

## Overview

This repository contains my solution to the **Spaceship Titanic** Kaggle competition. The goal of this competition is to predict whether a passenger was transported to an alternate dimension during the Spaceship Titanic's collision with the spacetime anomaly. The dataset provided includes information about the passengers, such as their cabin, age, destination, and more.

### Competition Link
[Spaceship Titanic on Kaggle](https://www.kaggle.com/competitions/spaceship-titanic)

### Dataset
The dataset can be downloaded from the competition page:
- [Train Data](https://www.kaggle.com/competitions/spaceship-titanic/data?select=train.csv)
- [Test Data](https://www.kaggle.com/competitions/spaceship-titanic/data?select=test.csv)

---

## Project Structure
spaceship-titanic/
├── input/ # Folder containing dataset files
│ ├── train.csv # Training data
│ ├── test.csv # Test data
├── notebooks/ # Jupyter notebooks for EDA and modeling
│ ├── eda.ipynb # Exploratory Data Analysis
├── src/ # Source code for preprocessing and modeling
│ ├── preprocess.py # Data preprocessing script
│ ├── train.py # Model training script
  ├── config.py # Configurations
  ├── model_dispatcher.py # Model repo
  ├── run.sh # bash script to run training on different folds for different models
├── requirements.txt # Python dependencies
├── README.md # This file

---

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/spaceship-titanic.git
   cd spaceship-titanic

# Approach
## Data Preprocessing:

Handled missing values by imputing with mean/mode or using advanced techniques like KNN imputation.

Encoded categorical variables using one-hot encoding or label encoding.

Scaled numerical features for better model performance.

## Feature Engineering:

Extracted useful information from the Cabin column (e.g., deck, side).

Created new features like FamilySize and IsAlone based on passenger groups.

## Modeling:

Experimented with various machine learning models, including Logistic Regression, Random Forest, Gradient Boosting, and XGBoost.

Used cross-validation and hyperparameter tuning to optimize model performance.

## Evaluation:

Evaluated models using accuracy, precision, recall, and F1-score.

Generated a submission file for the Kaggle competition.