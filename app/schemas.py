from pydantic import BaseModel, EmailStr
from typing import Optional


class SymptomInput(BaseModel):
    symptoms: str

class DiagnosisOutput(BaseModel):
    diagnosis: str

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class ProfileUpdate(BaseModel):
    email: str  # To identify the user
    gender: Optional[str] = None
    age: Optional[int] = None
    blood_group: Optional[str] = None
    height: Optional[float] = None
    weight: Optional[float] = None
    history: Optional[str] = None