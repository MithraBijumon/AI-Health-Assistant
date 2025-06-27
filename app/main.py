from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import SymptomInput, DiagnosisOutput
from app.model import diagnose

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify frontend URL here instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/diagnose", response_model=DiagnosisOutput)
def get_diagnosis(data: SymptomInput):
    result = diagnose(data.symptoms)
    return DiagnosisOutput(diagnosis=result)