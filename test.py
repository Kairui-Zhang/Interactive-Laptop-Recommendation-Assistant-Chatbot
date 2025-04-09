# Standard library
import sqlite3

# ------------------------------------------------------------

DATABASE = 'chat_history.db'

def check_database():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    query = 'SELECT * FROM history'
    cursor.execute(query)
    rows = cursor.fetchall()
   
    for row in rows:
        print(row)
        
    conn.close()

check_database()