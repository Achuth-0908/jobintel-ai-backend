import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class JobMatcher:
    def __init__(self, job_data_path='data/jobs_descriptions.json', use_semantic=True):
        self.job_data_path = job_data_path
        self.jobs_df = None
        self.job_data = None

        self.tfidf_vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
        self.job_vectors = None

        self.use_semantic = use_semantic and SentenceTransformer is not None
        self.sentence_model = None
        self.job_embeddings = None

        self.load_job_data()
        if self.jobs_df is not None:
            self.setup_vectorizer()
            if self.use_semantic:
                self.setup_semantic_embeddings()

    # ------------------- Data Loading -------------------
    def load_job_data(self):
        """Load job data and flatten for analysis"""
        try:
            with open(self.job_data_path, 'r') as f:
                self.job_data = json.load(f)

            flattened_jobs = []
            for company_data in self.job_data:
                company_name = company_data.get('company', 'Unknown')
                for job in company_data.get('jobs', []):
                    flattened_jobs.append({
                        'company': company_name,
                        'title': job.get('title', ''),
                        'position': job.get('position', ''),
                        'description': job.get('description', ''),
                        'skills': [s.lower() for s in job.get('skills', [])],
                        'remote_friendly': company_data.get('remoteFriendly', False),
                        'location': job.get('location', ''),
                        'posted': job.get('posted', ''),
                        'salary_from': job.get('salaryRange', {}).get('from', 0),
                        'salary_to': job.get('salaryRange', {}).get('to', 0),
                        'job_url': job.get('url', ''),
                    })
            self.jobs_df = pd.DataFrame(flattened_jobs)
            print(f"[INFO] Loaded {len(self.jobs_df)} jobs from {len(self.job_data)} companies")
        except Exception as e:
            print(f"[ERROR] Loading job data failed: {e}")
            self.jobs_df = None

    # ------------------- Vectorization -------------------
    def setup_vectorizer(self):
        """Precompute TF-IDF vectors for all jobs"""
        job_texts = [
            f"{row['description']} {' '.join(row['skills'])} {row['position']} {row['title']}"
            for _, row in self.jobs_df.iterrows()
        ]
        self.job_vectors = self.tfidf_vectorizer.fit_transform(job_texts)
        print("[INFO] TF-IDF vectorization complete")

    def setup_semantic_embeddings(self):
        """Precompute embeddings for semantic search"""
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            job_texts = [
                f"{row['description']} {' '.join(row['skills'])}"
                for _, row in self.jobs_df.iterrows()
            ]
            self.job_embeddings = self.sentence_model.encode(job_texts, show_progress_bar=True)
            print("[INFO] Semantic embeddings precomputed")
        except Exception as e:
            print(f"[WARNING] Semantic embeddings setup failed: {e}")
            self.use_semantic = False

    # ------------------- Scoring -------------------
    @staticmethod
    def skill_match(resume_skills, job_skills):
        if not resume_skills or not job_skills:
            return 0.0
        resume_set = set([s.lower() for s in resume_skills])
        job_set = set([s.lower() for s in job_skills])
        return len(resume_set & job_set) / len(job_set)

    @staticmethod
    def recent_posting_boost(posted_date):
        try:
            post_dt = datetime.strptime(posted_date, "%Y-%m-%d")
            days_old = (datetime.now() - post_dt).days
            return 1.0 if days_old < 7 else max(0.5, 1 - days_old / 90)  # decays over 90 days
        except:
            return 0.8

    def compute_scores(self, resume_skills, resume_text):
        """Compute combined score for all jobs"""
        n_jobs = len(self.jobs_df)
        skill_scores = np.array([self.skill_match(resume_skills, row['skills']) for _, row in self.jobs_df.iterrows()])
        
        # TF-IDF similarity
        tfidf_scores = np.zeros(n_jobs)
        if self.job_vectors is not None:
            resume_vec = self.tfidf_vectorizer.transform([resume_text])
            tfidf_scores = cosine_similarity(resume_vec, self.job_vectors)[0]

        # Semantic similarity
        semantic_scores = np.zeros(n_jobs)
        if self.use_semantic and self.job_embeddings is not None:
            resume_emb = self.sentence_model.encode([resume_text])
            semantic_scores = cosine_similarity(resume_emb, self.job_embeddings)[0]

        # Recent posting boost
        posting_boosts = np.array([self.recent_posting_boost(row['posted']) for _, row in self.jobs_df.iterrows()])

        # Combined score with weights
        combined_scores = (
            0.5 * skill_scores +
            0.3 * tfidf_scores +
            0.2 * semantic_scores
        ) * posting_boosts

        return combined_scores, skill_scores, tfidf_scores, semantic_scores

    # ------------------- Job Recommendation -------------------
    def get_job_recommendations(self, resume_skills, resume_text, top_n=10):
        combined_scores, skill_scores, tfidf_scores, semantic_scores = self.compute_scores(resume_skills, resume_text)

        results = []
        for idx, score in enumerate(combined_scores):
            row = self.jobs_df.iloc[idx].to_dict()
            row['match_score'] = float(round(score * 100, 2))
            row['skill_match_score'] = float(round(skill_scores[idx] * 100, 2))
            row['text_similarity_score'] = float(round(tfidf_scores[idx] * 100, 2))
            row['semantic_similarity_score'] = float(round(semantic_scores[idx] * 100, 2))
            row['matched_skills'] = list(set([s.lower() for s in resume_skills]) & set([s.lower() for s in row['skills']]))
            row['matched_skills_count'] = len(row['matched_skills'])
            results.append(row)

        results.sort(key=lambda x: x['match_score'], reverse=True)

        # Generate recommendation reasons
        for job in results[:top_n]:
            reasons = []
            if job['skill_match_score'] > 30:
                reasons.append(f"Strong skill match ({job['skill_match_score']:.1f}%)")
            if job['matched_skills_count'] > 0:
                reasons.append(f"Matches {job['matched_skills_count']} key skills")
            if job['semantic_similarity_score'] > 20:
                reasons.append("Good semantic similarity with job description")
            if job['remote_friendly']:
                reasons.append("Remote-friendly position")
            if not reasons:
                reasons.append("Basic compatibility with your profile")
            job['recommendation_reasons'] = reasons

        return results[:top_n]
