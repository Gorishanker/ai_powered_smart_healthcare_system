# models.py
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    name: Optional[str] = None
    # birth_date: Optional[datetime] = None

    class Config:
        orm_mode = True

class UserOut(BaseModel):
    id: int
    username: str
    email: str
    name: Optional[str] = None
    # birth_date: Optional[datetime] = None

    class Config:
        orm_mode = True
