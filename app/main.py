from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import SymptomInput, DiagnosisOutput, UserCreate, UserLogin, ProfileUpdate, ProfileUpdate
from app.model import diagnose
from app.auth import register_user, login_user
from app.database import cursor, conn

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

@app.post("/signup")
def signup(user: UserCreate):
    return register_user(user.name, user.email, user.password)

@app.post("/login")
def login(user: UserLogin):
    return login_user(user.email, user.password)

@app.post("/update-profile")
def update_profile(profile: ProfileUpdate):
    cursor.execute("SELECT * FROM users WHERE email = ?", (profile.email,))
    user = cursor.fetchone()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    cursor.execute("""
        UPDATE users
        SET gender = ?, age = ?, blood_group = ?, height = ?, weight = ?, history = ?
        WHERE email = ?
    """, (
        profile.gender,
        profile.age,
        profile.blood_group,
        profile.height,
        profile.weight,
        profile.history,
        profile.email
    ))
    conn.commit()

    cursor.execute("SELECT * FROM users WHERE email = ?", (profile.email,))
    updated_user = cursor.fetchone()
    columns = [col[0] for col in cursor.description]
    user_dict = dict(zip(columns, updated_user))

    return {"updatedUser": user_dict}

