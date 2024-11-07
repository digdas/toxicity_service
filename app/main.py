from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.database import engine, SessionLocal, Base, get_db
from app.model import load_model, predict_toxicity
from app.schemas import TextRequest, TextResponse, TextData
from app import models

# Initialize the app and database
app = FastAPI()
models.Base.metadata.create_all(bind=engine)
model = load_model()

@app.post("/predict", response_model=TextResponse)
async def predict_toxicity_route(request: TextRequest, db: Session = Depends(get_db)):
    labels = predict_toxicity(model, request.text)
    db_text = TextData(text=request.text, labels=labels)
    db.add(db_text)
    db.commit()
    db.refresh(db_text)
    return db_text

# Serve the web interface
@app.get("/")
async def read_index():
    return templates.TemplateResponse("index.html", {"request": request})
