# JobIntel Backend

## Overview

*JobIntel Backend* is a robust, scalable RESTful API built with *Flask* that powers job-resume matching and advanced skill extraction. It leverages NLP techniques such as *TF-IDF vectorization, **sentence embeddings (via Sentence Transformers), and **spaCy/NLTK-based skill extraction* to semantically analyze resumes and job descriptions for effective talent-job alignment.

---

## Features

- TF-IDF Based Resume-to-Job Matching
- Semantic Matching with Sentence Embeddings
- Multi-Method Skill Extraction
  - Regex-based
  - spaCy-based NLP
  - Semantic similarity-based
- Skill categorization into predefined domains (e.g., Programming, Cloud, Web Dev, etc.)
- Memory-aware model loading for efficient operation on low-resource machines

---

## Tech Stack

- *Backend*: Flask, Flask-CORS
- *NLP Libraries*: spaCy, NLTK, Sentence Transformers
- *ML*: scikit-learn, TF-IDF, Cosine Similarity
- *Data*: JSON-based Job Dataset
- *Utilities*: psutil, re, gc, pandas, numpy

---

## Directory Structure


.
├── app/                    # Core Flask app directory
│   ├── job_matcher.py      # JobMatcher class (TF-IDF + Semantic)
│   ├── skill_extractor.py  # SkillExtractor class
│   └── routes.py           # API routes (if applicable)
├── data/
│   └── jobs_descriptions.json  # Job dataset
├── main.py                 # Entry point for the Flask app
├── requirements.txt        # Dependency list
└── README.md               # This file


---

## Setup Instructions

### 1. Clone the Repository

bash
git clone https://github.com/yourusername/jobintel-backend.git
cd jobintel-backend


### 2. Create a Virtual Environment

bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate


### 3. Install Dependencies

bash
pip install -r requirements.txt


> If spaCy throws a model loading error, download the model manually:

bash
python -m spacy download en_core_web_sm


### 4. Run the Flask App

bash
python main.py


---

## API Endpoints

> *Base URL:* http://localhost:5000/

| Endpoint           | Method | Description                                      |
|--------------------|--------|--------------------------------------------------|
| /match           | POST   | Match resume text to jobs (TF-IDF/Semantic)     |
| /extract-skills  | POST   | Extract and categorize skills from resume text  |
| /health          | GET    | Health check for backend status                 |

---

## JSON Input Format

### /match

json
{
  "resume_text": "Skilled in Python, Django, and REST APIs...",
  "method": "tfidf" // or "semantic"
}


### /extract-skills

json
{
  "text": "Experienced in AWS, Docker, and Kubernetes."
}


---

## Output Format

### /match Response

json
[
  {
    "job_id": "101",
    "title": "Machine Learning Engineer",
    "score": 0.8723,
    "description": "Looking for an ML engineer with Python and TensorFlow..."
  }
]


### /extract-skills Response

json
{
  "all_skills": ["aws", "docker", "kubernetes"],
  "categorized_skills": {
    "cloud_devops": ["aws", "docker", "kubernetes"],
    "programming": []
  },
  "skill_count": 3,
  "extraction_methods": {
    "regex": 3,
    "spacy": 2,
    "semantic": 1
  }
}


---

## Dataset Format

File: data/jobs_descriptions.json

json
[
  {
    "id": "101",
    "title": "Data Scientist",
    "description": "We are looking for a data scientist skilled in Python, Pandas..."
  }
]


---

## Performance Considerations

- Uses conditional model loading (spaCy/SentenceTransformer) based on available RAM
- TF-IDF vectorization is memory-optimized using max_features=500
- Garbage collection (gc) is manually triggered for memory cleanup

---

## Deployment (Optional)

To run in production:

bash
gunicorn main:app --bind 0.0.0.0:5000 --workers 4


---
