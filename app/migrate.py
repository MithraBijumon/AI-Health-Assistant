import sqlite3
from database import cursor, conn

# Create table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL
)
""")

# Safely try to add new columns
columns_to_add = [
    ("gender", "TEXT"),
    ("age", "INTEGER"),
    ("blood_group", "TEXT"),
    ("height", "REAL"),
    ("weight", "REAL"),
    ("history", "TEXT"),
]

for col_name, col_type in columns_to_add:
    try:
        cursor.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_type};")
        print(f"✅ Added column: {col_name}")
    except sqlite3.OperationalError as e:
        if f"duplicate column name: {col_name}" in str(e):
            print(f"⚠️ Column already exists: {col_name}")
        else:
            raise

conn.commit()
conn.close()
print("✅ Migration script completed.")

