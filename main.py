from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import pandas as pd
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, text, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, Session, relationship
import joblib
from auth import verify_password, get_password_hash
from security import create_access_token
from datetime import datetime

app = FastAPI()
DATABASE_URL = "mysql+pymysql://root:password@localhost/ai_diease_predication"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Define the Feedback table
class Feedback(Base):
    __tablename__ = 'feedback'

    id = Column(Integer, primary_key=True, index=True)
    symptoms = Column(String)
    predicted_disease = Column(String)
    corrected_disease = Column(String)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String, nullable=True)
    password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


# Load data from Excel file
data = pd.read_excel("symptom_disease_data.xlsx")
symptom_disease_map = {}

# Load the trained model and symptom encoder
model = joblib.load("disease_model.joblib")
symptom_binarizer = joblib.load("symptom_binarizer.joblib")

# Prepare the symptom-disease map from the Excel file
for _, row in data.iterrows():
    symptom = row['symptom'].strip().lower()
    disease = row['disease'].strip()
    if symptom in symptom_disease_map:
        symptom_disease_map[symptom].append(disease)
    else:
        symptom_disease_map[symptom] = [disease]

# Define the input schema
class SymptomInput(BaseModel):
    symptoms: list[str]

# Define the feedback schema
class FeedbackInput(BaseModel):
    symptoms: list[str]
    predicted_disease: str
    corrected_disease: str

@app.post("/predict")
async def predict_disease(input_data: SymptomInput):
    possible_diseases = {}
    unmatched_symptoms = []

    for symptom in input_data.symptoms:
        symptom = symptom.strip().lower()
        if symptom in symptom_disease_map:
            diseases = symptom_disease_map[symptom]
            if len(diseases) == 1:
                # If there’s only one disease for a symptom, label it as highly likely
                possible_diseases[symptom] = {
                    "likely_disease": diseases[0],
                    "certainty": "high"
                }
            else:
                # For multiple matches, return a list of possible diseases
                possible_diseases[symptom] = {
                    "possible_diseases": diseases,
                    "certainty": "low"
                }
        else:
            unmatched_symptoms.append(symptom)

    # Construct the final response
    response = {
        "disease_predictions": possible_diseases,
        "unmatched_symptoms": unmatched_symptoms
    }
    if not possible_diseases:
        response["message"] = "No matching diseases found for the symptoms provided."
    else:
        response["message"] = "Disease predictions based on provided symptoms."

    return response


class UpdateDataInput(BaseModel):
    symptom: str
    disease: str

@app.post("/update_data")
async def update_data(input_data: UpdateDataInput):
    # Load the current data from the Excel file
    data = pd.read_excel("symptom_disease_data.xlsx")

    # Check if the entry already exists
    if ((data['symptom'].str.lower() == input_data.symptom.lower()) & 
        (data['disease'].str.lower() == input_data.disease.lower())).any():
        raise HTTPException(status_code=400, detail="This symptom-disease pair already exists.")

    # Append new data and save it back to the Excel file
    new_data = pd.DataFrame([[input_data.symptom, input_data.disease]], columns=["symptom", "disease"])
    updated_data = pd.concat([data, new_data], ignore_index=True)
    updated_data.to_excel("symptom_disease_data.xlsx", index=False)

    # Refresh the in-memory map for updated predictions
    refresh_symptom_disease_map()
    return {"message": "Data updated successfully"}

# Function to refresh the in-memory symptom-disease map
def refresh_symptom_disease_map():
    global symptom_disease_map
    data = pd.read_excel("symptom_disease_data.xlsx")
    symptom_disease_map = {}
    for _, row in data.iterrows():
        symptom = row['symptom'].strip().lower()
        disease = row['disease'].strip()
        if symptom in symptom_disease_map:
            symptom_disease_map[symptom].append(disease)
        else:
            symptom_disease_map[symptom] = [disease]

@app.post("/predict_disease_ml")
async def predict_disease_ml(input_data: SymptomInput):
    # Encode symptoms using the symptom encoder
    symptoms = [symptom.strip().lower() for symptom in input_data.symptoms]
    symptom_vector = symptom_binarizer.transform([symptoms])

    # Predict the disease using the model
    try:
        prediction = model.predict(symptom_vector)
        predicted_disease = prediction[0]
        return {"predicted_disease": predicted_disease}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
@app.post("/feedback")
async def submit_feedback(feedback: FeedbackInput):
    # Store feedback in the database
    with engine.connect() as connection:
        symptoms_str = ', '.join(feedback.symptoms)
        query = text("""
            INSERT INTO feedback (symptoms, predicted_disease, corrected_disease)
            VALUES (:symptoms, :predicted_disease, :corrected_disease)
        """)
        connection.execute(query, {
            "symptoms": symptoms_str,
            "predicted_disease": feedback.predicted_disease,
            "corrected_disease": feedback.corrected_disease
        })
    return {"message": "Feedback recorded successfully."}


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Login route to get JWT Token
@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register")
async def register_user(name: str, email: str, password: str, db: Session = Depends(get_db)):
    user_exists = db.query(User).filter(User.email == email).first()
    if user_exists:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(password)
    new_user = User(full_name=name, username=email,  email=email, password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully"}