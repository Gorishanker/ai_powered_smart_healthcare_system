from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sqlalchemy import create_engine, text

DATABASE_URL = "mysql+pymysql://root:password@localhost/ai_diease_predication"
engine = create_engine(DATABASE_URL)

# Query to fetch data from the database
query = """
SELECT symptoms, corrected_disease
FROM feedback
WHERE symptoms IS NOT NULL AND symptoms != '' AND corrected_disease IS NOT NULL AND corrected_disease != '';
"""

# Use text() correctly here
with engine.connect() as connection:
    result = connection.execute(text(query))  # Ensure the query is wrapped with text()
    feedback_data = result.fetchall()

# Convert the fetched data into a DataFrame
feedback_df = pd.DataFrame(feedback_data, columns=['symptoms', 'corrected_disease'])

# Preprocess symptoms
feedback_df['symptoms'] = feedback_df['symptoms'].apply(lambda x: x.split(', ') if isinstance(x, str) else [x])

# Initialize MultiLabelBinarizer and fit the symptoms
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(feedback_df['symptoms'])
y = feedback_df['corrected_disease']

# Check if there's enough data to split
if len(feedback_df) > 1:
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')
else:
    # Use the single record to train (for demonstration purposes)
    model = RandomForestClassifier()
    model.fit(X, y)
    print(f'Trained on single record.')

# Save the trained model and encoder
joblib.dump(model, 'disease_predictor_model.pkl')
joblib.dump(mlb, 'mlb_encoder.pkl')
