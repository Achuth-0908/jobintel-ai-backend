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
        """Calculate skill match score between resume and job"""
        if not resume_skills or not job_skills:
            return 0.0
        
        resume_skills_lower = [skill.lower() for skill in resume_skills]
        job_skills_lower = [skill.lower() for skill in job_skills]
        
        matched_skills = set(resume_skills_lower) & set(job_skills_lower)
        
        if not job_skills_lower:
            return 0.0
        
        return len(matched_skills) / len(job_skills_lower)
    
    def find_matching_jobs(self, resume_skills, resume_text, top_n=10):
        """Find matching jobs based on skills and resume text"""
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

        combined_scores = []
        for i in range(len(self.jobs_df)):
            combined_score = (
                0.5 * skill_scores[i] +
                0.3 * text_similarities[i] +
                0.2 * semantic_similarities[i]
            )
            combined_scores.append(combined_score)

        results = []
        for idx, score in enumerate(combined_scores):
            job = self.jobs_df.iloc[idx].to_dict()
            job['match_score'] = float(round(score * 100, 2))
            job['skill_match_score'] = float(round(skill_scores[idx] * 100, 2))
            job['text_similarity_score'] = float(round(text_similarities[idx] * 100, 2))
            job['semantic_similarity_score'] = float(round(semantic_similarities[idx] * 100, 2))
            
            resume_skills_lower = [skill.lower() for skill in resume_skills]
            job_skills_lower = [skill.lower() for skill in job['skills']]
            matched_skills = list(set(resume_skills_lower) & set(job_skills_lower))
            job['matched_skills'] = matched_skills
            job['matched_skills_count'] = int(len(matched_skills))

            results.append(job)

        results.sort(key=lambda x: x['match_score'], reverse=True)
        return results[:top_n]
    
    def get_job_recommendations(self, resume_skills, resume_text, top_n=10):
        """Get job recommendations with detailed analysis"""
        matching_jobs = self.find_matching_jobs(resume_skills, resume_text, top_n)

        for job in matching_jobs:
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
        
        return matching_jobs