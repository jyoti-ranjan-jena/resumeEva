# Resume Relevance Checker

An **automated tool** to evaluate resumes against job descriptions, providing **relevance scores** and **suggestions** to improve alignment.

---

## 🚀 Features

* Upload a **resume** (PDF, DOCX, or TXT) and a **job description**.
* Automatically **compute relevance scores** for skills, experience, and keywords.
* Generate **actionable suggestions** to improve resume alignment.
* View **historical evaluations** with analytics and charts.
* Built with a **modern tech stack** for fast and interactive experience.

---

## 🎥 Demo Video

* Watch the demo video showcasing our project:
* Link : https://youtu.be/IP5pMx8AVyA
* Click the image or link above to view the full video.

---

## 🛠️ Tech Stack

* **Frontend:** React.js, Tailwind CSS, Plotly/Other graphing libraries
* **Backend:** Node.js, Express.js, Python (for scoring)
* **Database:** MongoDB
* **AI & NLP:** OpenAI, Chroma for semantic search and parsing

---

## 📁 Project Structure

```text
resume-relevance/
│
├── app.py                 # Main Streamlit interface
├── parser.py              # Resume & JD text extraction
├── scoring.py             # Score computation and suggestions
├── db.py                  # Database connection & CRUD operations
├── requirements.txt       # Python dependencies
├── requirements-advanced.txt # Optional advanced dependencies (Torch, etc.)
├── sample_resume.txt
├── sample_jd.txt
└── .gitignore             # Ignore venv, database, cache files
```

---

## ⚡ Installation

1. **Clone the repo:**

```bash
git clone https://github.com/jyoti-ranjan-jena/resume-relevance-check.git
cd resume-relevance-check
```

2. **Create a virtual environment and install dependencies:**

```bash
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

3. **Run the app:**

```bash
streamlit run app.py
```

4. Open the app in your browser: [http://localhost:8501](http://localhost:8501)

---

## 🔧 Usage

1. Upload your **resume** and **job description** files.
2. Click **Evaluate**.
3. View **scores** and **suggestions**.
4. Optionally, save evaluations to the database for future reference.

---

## ⚠️ Notes

* The `venv/` folder and other heavy files like `.dll` and `.lib` are **not included** in the repository.
* Keep your Python environment **isolated** to avoid pushing large files.

---

## 📂 Contributions

Contributions are welcome!

* Fork the repo
* Create a new branch (`git checkout -b feature/your-feature`)
* Commit your changes (`git commit -m 'Add new feature'`)
* Push (`git push origin feature/your-feature`)
* Open a pull request

---
