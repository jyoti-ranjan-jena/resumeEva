# db.py
import sqlite3
import datetime
import json

DB_PATH = "evaluations.db"

def init_db(path=DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        resume_filename TEXT,
        job_title TEXT,
        score INTEGER,
        verdict TEXT,
        missing_skills TEXT,
        matched_skills TEXT,
        resume_snippet TEXT,
        jd_text TEXT
    )""")
    conn.commit()
    return conn

def save_evaluation(conn, resume_filename, job_title, score, verdict, missing_skills, matched_skills, resume_snippet, jd_text):
    c = conn.cursor()
    c.execute("""
      INSERT INTO evaluations (timestamp, resume_filename, job_title, score, verdict, missing_skills, matched_skills, resume_snippet, jd_text)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
      (datetime.datetime.utcnow().isoformat(), resume_filename, job_title, int(score), verdict,
       json.dumps(missing_skills), json.dumps(matched_skills), resume_snippet[:1000], jd_text[:2000])
    )
    conn.commit()

def fetch_evaluations(conn, min_score=0, job_title_substr=None):
    c = conn.cursor()
    if job_title_substr:
        q = "SELECT * FROM evaluations WHERE score >= ? AND job_title LIKE ? ORDER BY score DESC"
        rows = c.execute(q, (min_score, f"%{job_title_substr}%")).fetchall()
    else:
        q = "SELECT * FROM evaluations WHERE score >= ? ORDER BY score DESC"
        rows = c.execute(q, (min_score,)).fetchall()
    cols = [d[0] for d in c.description]
    return [dict(zip(cols, r)) for r in rows]
