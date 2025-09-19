import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

class JobMatcher:
    def __init__(self, job_data_path='data/jobs_descriptions.json'):
        self.job_data_path = job_data_path
        self.jobs_df = None
        self.job_data = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        self.sentence_model = None
        if SentenceTransformer is not None:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                print("Could not load sentence transformer model")
                self.sentence_model = None
        
        self.job_vectors = None
        self.load_job_data()
        if self.jobs_df is not None:
            self.setup_vectorizer()
    
    def load_job_data(self):
        """Load job descriptions from JSON file"""
        try:
            with open(self.job_data_path, 'r') as f:
                self.job_data = json.load(f)

            flattened_jobs = []
            for company_data in self.job_data:
                company_name = company_data.get('company', 'Unknown')
                company_url = company_data.get('url', '')
                remote_friendly = company_data.get('remoteFriendly', False)
                market = company_data.get('market', '')
                size = company_data.get('size', '')
                
                for job in company_data.get('jobs', []):
                    job_record = {
                        'company': company_name,
                        'company_url': company_url,
                        'remote_friendly': remote_friendly,
                        'market': market,
                        'company_size': size,
                        'position': job.get('position', ''),
                        'title': job.get('title', ''),
                        'description': job.get('description', ''),
                        'job_url': job.get('url', ''),
                        'type': job.get('type', ''),
                        'posted': job.get('posted', ''),
                        'location': job.get('location', ''),
                        'skills': job.get('skills', []),
                        'salary_from': job.get('salaryRange', {}).get('from', 0),
                        'salary_to': job.get('salaryRange', {}).get('to', 0),
                        'currency': job.get('salaryRange', {}).get('currency', 'USD'),
                        'equity_from': job.get('equity', {}).get('from', 0),
                        'equity_to': job.get('equity', {}).get('to', 0),
                        'perks': job.get('perks', []),
                        'apply_url': job.get('apply', '')
                    }
                    flattened_jobs.append(job_record)
            
            self.jobs_df = pd.DataFrame(flattened_jobs)
            print(f"Loaded {len(self.jobs_df)} jobs from {len(self.job_data)} companies")
            
        except FileNotFoundError:
            print(f"Job data file not found: {self.job_data_path}")
            self.jobs_df = None
        except Exception as e:
            print(f"Error loading job data: {e}")
            self.jobs_df = None
    
    def setup_vectorizer(self):
        """Setup TF-IDF vectorizer with job descriptions"""
        if self.jobs_df is None:
            return
        
        try:
            job_texts = []
            for _, row in self.jobs_df.iterrows():
                skills_text = ' '.join(row['skills']) if row['skills'] else ''
                combined_text = f"{row['description']} {skills_text} {row['position']} {row['title']}"
                job_texts.append(combined_text)
            
            self.job_vectors = self.tfidf_vectorizer.fit_transform(job_texts)
            print("TF-IDF vectorizer setup completed")
        except Exception as e:
            print(f"Error setting up vectorizer: {e}")
    
    def calculate_skill_match_score(self, resume_skills, job_skills):
        """Calculate skill match score between resume and job (Jaccard similarity)"""
        if not resume_skills or not job_skills:
            return 0.0
        resume_skills_lower = set([skill.lower() for skill in resume_skills])
        job_skills_lower = set([skill.lower() for skill in job_skills])
        intersection = resume_skills_lower & job_skills_lower
        union = resume_skills_lower | job_skills_lower
        if not union:
            return 0.0
        return len(intersection) / len(union)

    def normalize_scores(self, scores):
        """Min-max normalize a list of scores to [0, 1]"""
        if not scores:
            return scores
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [0.0 for _ in scores]
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def find_matching_jobs(self, resume_skills, resume_text):
        """Find matching jobs based on skills and resume text (return all jobs)"""
        if self.jobs_df is None or len(self.jobs_df) == 0:
            return []
        if not resume_skills:
            return []

        skill_scores = []
        for idx, row in self.jobs_df.iterrows():
            skill_score = self.calculate_skill_match_score(resume_skills, row['skills'])
            skill_scores.append(skill_score)

        text_similarities = []
        if self.job_vectors is not None:
            try:
                resume_vector = self.tfidf_vectorizer.transform([resume_text])
                text_similarities = cosine_similarity(resume_vector, self.job_vectors)[0]
            except Exception as e:
                print(f"Error in TF-IDF similarity: {e}")
                text_similarities = [0.0] * len(self.jobs_df)
        else:
            text_similarities = [0.0] * len(self.jobs_df)

        semantic_similarities = []
        if self.sentence_model:
            try:
                resume_embedding = self.sentence_model.encode([resume_text])
                job_texts = []
                for _, row in self.jobs_df.iterrows():
                    skills_text = ' '.join(row['skills']) if row['skills'] else ''
                    combined_text = f"{row['description']} {skills_text}"
                    job_texts.append(combined_text)
                job_embeddings = self.sentence_model.encode(job_texts)
                semantic_similarities = cosine_similarity(resume_embedding, job_embeddings)[0]
            except Exception as e:
                print(f"Error in semantic similarity: {e}")
                semantic_similarities = [0.0] * len(self.jobs_df)
        else:
            semantic_similarities = [0.0] * len(self.jobs_df)

        # Normalize scores for fair combination
        skill_scores_norm = self.normalize_scores(skill_scores)
        text_similarities_norm = self.normalize_scores(text_similarities)
        semantic_similarities_norm = self.normalize_scores(semantic_similarities)

        combined_scores = []
        for i in range(len(self.jobs_df)):
            # Adjust weights for better accuracy
            combined_score = (
                0.4 * skill_scores_norm[i] +
                0.3 * text_similarities_norm[i] +
                0.3 * semantic_similarities_norm[i]
            )
            combined_scores.append(combined_score)

        results = []
        for idx, score in enumerate(combined_scores):
            job = self.jobs_df.iloc[idx].to_dict()
            job['match_score'] = float(round(score * 100, 2))
            job['skill_match_score'] = float(round(skill_scores_norm[idx] * 100, 2))
            job['text_similarity_score'] = float(round(text_similarities_norm[idx] * 100, 2))
            job['semantic_similarity_score'] = float(round(semantic_similarities_norm[idx] * 100, 2))

            resume_skills_lower = set([skill.lower() for skill in resume_skills])
            job_skills_lower = set([skill.lower() for skill in job['skills']])
            matched_skills = list(resume_skills_lower & job_skills_lower)
            job['matched_skills'] = matched_skills
            job['matched_skills_count'] = int(len(matched_skills))

            results.append(job)

        results.sort(key=lambda x: x['match_score'], reverse=True)
        return results  # Return all jobs, sorted

    def get_job_recommendations(self, resume_skills, resume_text):
        """Get job recommendations with detailed analysis (all jobs, sorted)"""
        matching_jobs = self.find_matching_jobs(resume_skills, resume_text)

        for job in matching_jobs:
            reasons = []
            if job['skill_match_score'] > 50:
                reasons.append(f"Excellent skill match ({job['skill_match_score']:.1f}%)")
            elif job['skill_match_score'] > 30:
                reasons.append(f"Strong skill match ({job['skill_match_score']:.1f}%)")
            if job['matched_skills_count'] > 0:
                reasons.append(f"Matches {job['matched_skills_count']} key skills")
            if job['semantic_similarity_score'] > 40:
                reasons.append("High semantic similarity with job description")
            elif job['semantic_similarity_score'] > 20:
                reasons.append("Good semantic similarity with job description")
            if job.get('remote_friendly', False):
                reasons.append("Remote-friendly position")
            if not reasons:
                reasons.append("Basic compatibility with your profile")
            job['recommendation_reasons'] = reasons

        return matching_jobs