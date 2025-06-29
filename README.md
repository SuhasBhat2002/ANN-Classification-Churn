# 🔍 ANN Classification - Customer Churn Prediction

This project builds and deploys an Artificial Neural Network (ANN) to predict customer churn using bank data. It features a complete ML pipeline — from data preprocessing and model training to a live prediction interface via a Streamlit app.

---

## 📁 Project Structure

```
ANN-Classification-Churn/
├── app.py                  # Streamlit app for making predictions
├── churn_model.pth         # Trained PyTorch model
├── Churn_Modelling.csv     # Dataset used for training/testing
├── experiments.ipynb       # Model training and evaluation notebook
├── preprocessor.pkl        # Saved preprocessing pipeline
├── preprocessor.py         # Feature transformation logic
├── requirements.txt        # Project dependencies
```

---

## 🧠 Model Overview

* **Model Type**: Feedforward Neural Network (ANN)
* **Framework**: PyTorch
* **Architecture**:

  * Input layer: Processed features from customer data
  * Hidden layers: Fully connected with ReLU activations
  * Output: Sigmoid for binary churn prediction
* **Loss Function**: Binary Cross Entropy with **class weights** to address imbalance
* **Optimizer**: Adam

---

## 🧪 Jupyter Notebook: `experiments.ipynb`

This notebook handles:

* Data loading and exploration
* Feature selection and preprocessing
* Class imbalance handling:

  * ✅ **Class Weights**: Adjusted loss function to penalize the majority class less
  * ✅ **SMOTE**: Synthetic oversampling of the minority class to balance the dataset
* ANN model construction and training using PyTorch
* Performance evaluation using accuracy and confusion matrix

---
## 🌐 Streamlit App: `app.py`

Provides a user-friendly web interface to:

* Input customer attributes
* Load and apply the trained model and preprocessor
* Display churn prediction instantly

### Sample Usage

```bash
streamlit run app.py
```

---

## 📦 Installation & Setup

1. **Clone the repo**

```bash
git clone https://github.com/SuhasBhat2002/ANN-Classification-Churn.git
cd ANN-Classification-Churn
```

2. **Create a virtual environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the app**

```bash
streamlit run app.py
```

---

## 📊 Dataset Info

The dataset (`Churn_Modelling.csv`) includes:

* Customer demographics (age, gender, geography)
* Account info (credit score, balance, tenure)
* Churn label: whether the customer left the bank (0 or 1)

---

## 📈 Performance

Model performance metrics:

* Evaluated on a held-out test set
* Metrics include accuracy, confusion matrix, and recall
* Handled class imbalance with **SMOTE** and **weighted loss function**

---

## 🔮 Possible Improvements

* Add cross-validation and hyperparameter tuning
* Experiment with deeper network architectures
* Integrate a database or REST API backend for real-world deployment

---

## 📌 Dependencies

See `requirements.txt` for full list. Key libraries include:

* `pandas`, `numpy`
* `torch`, `sklearn`, `imblearn`
* `joblib`, `streamlit`

---

## ✍️ Author

**Suhas Bhat**
[GitHub Profile](https://github.com/SuhasBhat2002)

---


