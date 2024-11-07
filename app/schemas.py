from pydantic import BaseModel
from app.database import Base
from typing import List, Dict
from sqlalchemy import Column, Integer, String, JSON

class TextRequest(BaseModel):
    text: str

class TextResponse(BaseModel):
    text: str
    labels_with_scores: List[dict]

    class Config:
        orm_mode = True

class TextDataResponse(BaseModel):
    id: int
    text: str
    labels_with_scores: List[dict]

    class Config:
        orm_mode = True  # This tells Pydantic to treat the database model as a dictionary

class TextData(Base):
    __tablename__ = 'texts'
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    labels_with_scores = Column(JSON, nullable=False)  # Store labels and scores as a JSON column

    def __repr__(self):
        return f"<TextData(id={self.id}, text={self.text}, labels_with_scores={self.labels_with_scores})>"