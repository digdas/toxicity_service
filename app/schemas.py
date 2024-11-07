from pydantic import BaseModel
from app.database import Base
from sqlalchemy import Column, Integer, String, JSON

class TextRequest(BaseModel):
    text: str

class TextResponse(BaseModel):
    text: str
    labels: list[str]

class TextData(Base):
    __tablename__ = 'texts'
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    labels = Column(JSON, nullable=False)
