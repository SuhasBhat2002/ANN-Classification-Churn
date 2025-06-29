import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import pickle

# Load the preprocessor
from preprocessor import identity_func  # make sure this is imported BEFORE loading
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# Define model architecture (same as training)
class ChurnNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# Streamlit UI
st.title("Customer Churn Prediction")

credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=650)
geography = st.selectbox("Geography", ['France', 'Spain', 'Germany'])
gender = st.selectbox("Gender", ['Male', 'Female'])
age = st.slider("Age", min_value=18, max_value=100, value=35)
tenure = st.slider("Tenure", min_value=0, max_value=10, value=5)
balance = st.slider("Balance", min_value=0.0, max_value=250000.0, value=100000.0, step=1000.0)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.slider("Estimated Salary", min_value=0.0, max_value=200000.0, value=50000.0, step=1000.0)


# Prepare input as DataFrame
user_input = pd.DataFrame([{
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active,
    'EstimatedSalary': estimated_salary
}])

# Preprocess input
processed_input = preprocessor.transform(user_input)
input_tensor = torch.tensor(processed_input, dtype=torch.float32)

# Load model
input_dim = processed_input.shape[1]
model = ChurnNet(input_dim)
model.load_state_dict(torch.load("churn_model.pth", map_location=torch.device('cpu')))
model.eval()

# Make prediction
with torch.no_grad():
    output = model(input_tensor)
    prob = torch.sigmoid(output).item()
    result = "Churn" if prob >= 0.5 else "No Churn"

# Display
st.subheader("Prediction:")
st.write(f"**Probability of churn:** {prob:.2f}")
st.write(f"**Predicted class:** {result}")
