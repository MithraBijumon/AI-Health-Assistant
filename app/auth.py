import sqlite3
from fastapi import HTTPException
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Connect and ensure table exists
from app.database import cursor, conn

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL
)
""")
conn.commit()

def get_user_by_email(email: str):
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    return cursor.fetchone()

def register_user(name: str, email: str, password: str):
    if get_user_by_email(email):
        raise HTTPException(status_code=400, detail="Email already registered.")
    
    hashed_password = pwd_context.hash(password)
    cursor.execute("INSERT INTO users (name, email, hashed_password) VALUES (?, ?, ?)",
                   (name, email, hashed_password))
    conn.commit()
    return {"message": "Signup successful!"}

def login_user(email: str, password: str):
    user = get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    
    if not pwd_context.verify(password, user[3]):
        raise HTTPException(status_code=401, detail="Invalid password.")
    
    columns = [desc[0] for desc in cursor.description]
    user_dict = dict(zip(columns, user))
    
    return user_dict
