'''from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
import pandas as pd
import pickle

# Feature groups
numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
categorical_features = ['Geography', 'Gender']
binary_features = ['HasCrCard', 'IsActiveMember']'''

# Identity function for binary features
def identity_func(x):
    return x

'''# Define the preprocessing pipeline
pipeline = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(), categorical_features),
    ('bin', FunctionTransformer(identity_func), binary_features)
])

# Dummy data to fit the pipeline (same structure as user input)
dummy_data = pd.DataFrame({
    'CreditScore': [600, 700],
    'Geography': ['France', 'Germany'],
    'Gender': ['Male', 'Female'],
    'Age': [30, 40],
    'Tenure': [3, 6],
    'Balance': [50000.0, 75000.0],
    'NumOfProducts': [1, 2],
    'HasCrCard': [1, 0],
    'IsActiveMember': [1, 0],
    'EstimatedSalary': [40000.0, 65000.0]
})

# Fit the pipeline
pipeline.fit(dummy_data)

# Save the fitted pipeline
with open("preprocessor.pkl", "wb") as f:
    pickle.dump(pipeline, f)
'''