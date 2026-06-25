# ❤️ Heart Disease Prediction System

A Machine Learning web application that predicts the risk of heart disease based on patient health parameters. The model is trained on the Heart Disease dataset and deployed using Streamlit.

## 🚀 Project Overview

Heart disease is one of the leading causes of death worldwide. This project uses Machine Learning algorithms to predict whether a patient is at risk of heart disease based on medical attributes such as age, cholesterol level, blood pressure, chest pain type, ECG results, and more.

The application allows users to enter patient details through a simple Streamlit interface and receive an instant prediction.

---

## 📊 Dataset

Dataset Source:

- Kaggle Heart Disease Dataset

Downloaded using:

```python
import kagglehub

path = kagglehub.dataset_download(
    "amirmahdiabbootalebi/heart-disease"
)
```

### Features Used

| Feature | Description |
|----------|------------|
| Age | Age of patient |
| Sex | Male/Female |
| ChestPainType | Type of chest pain |
| RestingBP | Resting blood pressure |
| Cholesterol | Serum cholesterol |
| FastingBS | Fasting blood sugar |
| RestingECG | Resting electrocardiogram results |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Exercise induced angina |
| Oldpeak | ST depression induced by exercise |
| ST_Slope | Slope of peak exercise ST segment |
| HeartDisease | Target Variable |

---

## 🔍 Exploratory Data Analysis (EDA)

The following analyses were performed:

- Missing value inspection
- Cholesterol value correction
- Countplots for categorical variables
- Boxplots for numerical variables
- Violin plots for distribution analysis
- Correlation heatmap
- Target variable distribution analysis

### Data Cleaning

Invalid cholesterol values represented by 0 were replaced with the mean cholesterol value calculated from valid observations.

```python
cholesterol_mean = df.loc[
    df['Cholesterol'] != 0,
    'Cholesterol'
].mean()

df['Cholesterol'] = (
    df['Cholesterol']
    .replace(0, cholesterol_mean)
    .round(2)
)
```

---

## ⚙️ Data Preprocessing

### Categorical Encoding

One-Hot Encoding was applied using:

```python
pd.get_dummies(df, drop_first=True)
```

### Feature Scaling

StandardScaler was used to normalize feature values before training.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## 🤖 Machine Learning Models Used

The following classification algorithms were trained and evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Decision Tree Classifier
- Support Vector Machine (RBF Kernel)

```python
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM (RBF Kernel)": SVC(probability=True)
}
```

---

## 📈 Evaluation Metrics

Models were evaluated using:

- Accuracy Score
- F1 Score

```python
accuracy_score(y_test, y_pred)
f1_score(y_test, y_pred)
```

The best-performing model was selected and saved using Joblib.

---

## 💾 Model Saving

```python
import joblib

joblib.dump(model, "knn_heart_model.pkl")
joblib.dump(scaler, "heart_scaler.pkl")
joblib.dump(X.columns.tolist(), "heart_columns.pkl")
```

---

## 🌐 Streamlit Web Application

The Streamlit application allows users to:

- Enter patient information
- Perform automatic preprocessing
- Apply scaling
- Predict heart disease risk
- Display prediction results instantly

Run locally using:

```bash
streamlit run app.py
```

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Joblib
- Streamlit
- Git
- GitHub

---

## 📂 Project Structure

```text
Heart-Disease-Prediction/
│
├── app.py
├── knn_heart.pkl
├── scaler.pkl
├── columns.pkl
├── requirements.txt
├── .gitignore
├── Heart Disease Prediction.ipynb
└── README.md
```

---

## ⚡ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
```

### Navigate to Project

```bash
cd heart-disease-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 🎯 Prediction Output

### Low Risk

```text
✅ Low Risk of Heart Disease
```

### High Risk

```text
⚠️ High Risk of Heart Disease
```

---

## 📚 Learning Outcomes

Through this project, I learned:

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- One-Hot Encoding
- Feature Scaling
- Model Training
- Model Evaluation
- Model Serialization using Joblib
- Streamlit Deployment
- Git and GitHub Workflow

---

## 👨‍💻 Author

**Abhideepta**

MCA Student | Data Science & Machine Learning Enthusiast

---

## ⭐ If you found this project useful, consider giving it a star.
