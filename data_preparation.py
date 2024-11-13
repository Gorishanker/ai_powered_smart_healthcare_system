import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from joblib import dump


# Load data
data = pd.read_excel("symptom_disease_data.xlsx")

# Group symptoms by disease and create a list of symptom lists
grouped = data.groupby('disease')['symptom'].apply(list).reset_index()

# Convert symptoms to a binary matrix
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(grouped['symptom'])
y = grouped['disease']

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the model and the MultiLabelBinarizer
dump(model, "disease_model.joblib")
dump(mlb, "symptom_binarizer.joblib")

print("Model training complete and saved!")
