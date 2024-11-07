from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.database import engine, SessionLocal, Base, get_db
from app.model import load_model, predict_toxicity
from app.schemas import TextRequest, TextResponse, TextData, TextDataResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from typing import List

# Initialize the app and database
app = FastAPI()
Base.metadata.create_all(bind=engine)
model = load_model()
templates = Jinja2Templates(directory="app/web/templates")

@app.get("/predictions", response_model=List[TextDataResponse])
async def get_saved_predictions(db: Session = Depends(get_db)):
    # Fetch all saved predictions from the database
    saved_predictions = db.query(TextData).all()
    
    # Return the saved predictions
    return saved_predictions

@app.post("/predict", response_model=TextResponse)
async def predict_toxicity_route(request: TextRequest, db: Session = Depends(get_db)):
    labels_with_scores = predict_toxicity(model, request.text)
    
    # Save the text and prediction to the database
    db_text = TextData(text=request.text, labels_with_scores=labels_with_scores)
    db.add(db_text)
    db.commit()
    db.refresh(db_text)
    
    # Return the response
    return TextResponse(text=request.text, labels_with_scores=labels_with_scores)

# Serve the web interface
@app.get("/")
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
