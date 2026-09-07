# SpamShield

SpamShield is an SMS spam detection application built with Python and machine learning. It compares Naive Bayes, Logistic Regression, and Linear SVM models, then uses a tuned Linear SVM in the Streamlit application.

## Project Structure

```text
phase3_prjct_team5/
├── app.py
├── main.py
├── sanity_checks.py
├── data/
│   └── spam_dataset.csv
├── models/
├── notebooks/
│   └── exploration.ipynb
└── src/
    ├── preprocess.py
    ├── features.py
    ├── train.py
    └── evaluate.py
```

## Requirements

- Python 3.9 or later
- pandas
- numpy
- scikit-learn
- matplotlib
- joblib
- streamlit

## Installation

Open PowerShell in the project directory:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install pandas numpy scikit-learn matplotlib joblib streamlit
```

## Train the Models

Make sure the dataset is located at:

```text
data/spam_dataset.csv
```

Run:

```powershell
python .\main.py
```

This will:

1. Load and clean the dataset.
2. Split the data into training and testing sets.
3. Generate TF-IDF features.
4. Train three machine-learning models.
5. Tune the Linear SVM `C` parameter.
6. Display evaluation results.
7. Save the tuned model to:

```text
models/spam_model.joblib
```

## Run the Streamlit Application

After training, start the application:

```powershell
streamlit run .\app.py
```

Open the displayed local URL in a browser, enter an SMS message, and select **Check Message**.

## Models

The project evaluates:

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine
- Tuned Linear Support Vector Machine

The Streamlit app uses the tuned Linear SVM model.

## Evaluation

The following metrics are displayed:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report

The application also displays the held-out test accuracy.

## Notes

Run `main.py` again whenever the dataset or preprocessing code changes so that the saved model is updated.


## 🚀 Live Demo
You can try the live application here: [SpamShield Live App]https://optimized-spam-detector-iurswlvyfzgnnupumycjhe.streamlit.app/)


## 👥 Team Members & Responsibilities (Team 5)

| Member | Role | Key Responsibilities |
| :--- | :--- | :--- |
| **Mariam Tamer Mohed Elnady** | Data Engineering | Data collection, cleaning, preprocessing, and EDA |
| **Israa Eldsouky** | ML Engineer (Modeling) | Feature extraction (TF-IDF/Embeddings), baseline model implementations |
| **Nour Hussin Abdo Elafifi**   | ML Engineer (Optimization) | Hyperparameter tuning, cross-validation, and performance evaluation |
| **Malak Zaky Eid Zaky El-Santry** |  Project Leader & Deployment & Documentation | UI/App deployment (Streamlit/Flask), README, and final reporting |

---

### Detailed Task Distribution:

* **Mariam Tamer Mohed Elnady**:
  * Conducted dataset acquisition, data cleaning, handling missing values, and exploratory data analysis (EDA).
  * Built data preprocessing and tokenization pipelines.

* **Israa Eldsouky**:
  * Implemented text vectorization and feature engineering (e.g., TF-IDF / Bag of Words).
  * Developed and trained baseline classification models (e.g., Naive Bayes, Logistic Regression).
  * Performed training/testing split with stratified sampling.

* **Nour Hussin Abdo Elafifi**:
  * Conducted hyperparameter tuning and model optimization (e.g., SVM Grid Search, Cross-Validation).
  * Evaluated models using accuracy, precision, recall, F1-score, and confusion matrices.
  * Extracted diagnostic error analysis and misclassification insights.

* **Malak Zaky Eid Zaky El-Santry**:
  * Managed project roadmap and team coordination.
  * Built the interactive user interface and deployed the model application.
  * Formatted pipeline serialization (`joblib` / `pickle`) and caching for real-time inference.
  * Authored the project documentation, technical report, and `README.md`.
