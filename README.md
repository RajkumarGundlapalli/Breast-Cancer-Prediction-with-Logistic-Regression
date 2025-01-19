# Breast Cancer Prediction with Logistic Regression

This project utilizes logistic regression to predict whether a breast cancer tumor is malignant or benign. The project consists of two main components:

1. **Model Training**: A Python script to train a logistic regression model using breast cancer data and save the model and scaler.
2. **Streamlit Application**: A web application to interactively input data and get predictions from the trained model.

## Model Training

### Prerequisites

- Python 3.x
- Libraries: `pandas`, `seaborn`, `scikit-learn`, `pickle`

### Steps

1. **Install Dependencies**

   ```bash
   pip install pandas seaborn scikit-learn
   ```

2. **Training Script**

   The script performs the following:
   - Reads the dataset from `data.csv`.
   - Preprocesses the data (removes unnecessary columns, encodes the target variable).
   - Scales the features using `StandardScaler`.
   - Splits the data into training and test sets.
   - Trains a logistic regression model.
   - Evaluates the model and prints accuracy and classification report.
   - Saves the trained model and scaler as `model.pkl` and `scaler.pkl`.


3. **Run the Training Script**

   ```bash
   python train_model.py
   ```

## Streamlit Application

### Prerequisites

- Python 3.x
- Libraries: `streamlit`, `pandas`, `pickle`, `scikit-learn`

### Steps

1. **Install Dependencies**

   ```bash
   pip install streamlit pandas scikit-learn
   ```

2. **Streamlit Application**

   The application allows users to input feature values and receive a prediction of whether the tumor is malignant or benign.

3. **Run the Streamlit Application**

   ```bash
   streamlit run app.py



