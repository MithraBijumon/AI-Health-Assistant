from pydantic import BaseModel

class SymptomInput(BaseModel):
    symptoms: str

class DiagnosisOutput(BaseModel):
    diagnosis: str
