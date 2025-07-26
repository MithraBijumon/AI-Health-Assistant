import sqlite3

conn = sqlite3.connect("C:/Mithra/WNCC/health-assistant/app/users.db", check_same_thread=False)
cursor = conn.cursor()
