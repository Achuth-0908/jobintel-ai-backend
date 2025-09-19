import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import re
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer  # pyright: ignore[reportMissingImports]
except ImportError:
    SentenceTransformer = None

class JobMatcher:
    def __init__(self, job_data_path='data/jobs_descriptions.json', use_semantic=True):
        self.job_data_path = job_data_path
        self.jobs_df = None
        self.job_data = None
        
        # Optimized TF-IDF with better parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1500,  # Reduced from 2000 for faster computation
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),  # Include bigrams for better matching
            min_df=2,  # Remove very rare terms
            max_df=0.95  # Remove very common terms
        )
        self.job_vectors = None
        
        self.use_semantic = use_semantic and SentenceTransformer is not None
        self.sentence_model = None
        self.job_embeddings = None
        
        # Cache for preprocessed data
        self.skill_sets_cache = {}
        self.posting_boost_cache = {}
        self.job_texts_cache = None
        
        self.load_job_data()
        if self.jobs_df is not None:
            self._preprocess_data()
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
                remote_friendly = company_data.get('remoteFriendly', False)
                
                for job in company_data.get('jobs', []):
                    # Pre-process skills to lowercase once
                    skills = [s.lower().strip() for s in job.get('skills', []) if s.strip()]
                    
                    flattened_jobs.append({
                        'company': company_name,
                        'title': job.get('title', ''),
                        'position': job.get('position', ''),
                        'description': job.get('description', ''),
                        'skills': skills,
                        'remote_friendly': remote_friendly,
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

    def _preprocess_data(self):
        """Preprocess and cache data for faster access"""
        print("[INFO] Preprocessing data for optimization...")
        
        # Cache skill sets as frozensets for O(1) intersection
        for idx, skills in enumerate(self.jobs_df['skills']):
            self.skill_sets_cache[idx] = frozenset(skills)
        
        # Pre-calculate posting boosts
        for idx, posted_date in enumerate(self.jobs_df['posted']):
            self.posting_boost_cache[idx] = self._calculate_posting_boost(posted_date)
        
        # Pre-create job texts for vectorization
        self.job_texts_cache = [
            self._create_job_text(row) 
            for _, row in self.jobs_df.iterrows()
        ]
        
        print("[INFO] Data preprocessing complete")

    @staticmethod
    def _create_job_text(row):
        """Create optimized job text representation"""
        # Clean and join text components
        components = []
        
        if row['description']:
            # Simple text cleaning
            desc = re.sub(r'[^\w\s]', ' ', row['description'].lower())
            desc = ' '.join(desc.split())  # Remove extra whitespace
            components.append(desc)
        
        if row['skills']:
            components.append(' '.join(row['skills']))
        
        if row['position']:
            components.append(row['position'].lower())
        
        if row['title']:
            components.append(row['title'].lower())
            
        return ' '.join(components)

    @staticmethod
    def _calculate_posting_boost(posted_date):
        """Calculate posting boost with caching"""
        try:
            post_dt = datetime.strptime(posted_date, "%Y-%m-%d")
            days_old = (datetime.now() - post_dt).days
            return 1.0 if days_old < 7 else max(0.5, 1 - days_old / 90)
        except:
            return 0.8

    # ------------------- Vectorization -------------------
    def setup_vectorizer(self):
        """Precompute TF-IDF vectors for all jobs with optimization"""
        self.job_vectors = self.tfidf_vectorizer.fit_transform(self.job_texts_cache)
        print(f"[INFO] TF-IDF vectorization complete - Shape: {self.job_vectors.shape}")

    def setup_semantic_embeddings(self):
        """Precompute embeddings for semantic search"""
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Use only description and skills for semantic embeddings
            semantic_texts = [
                f"{row['description']} {' '.join(row['skills'])}"
                for _, row in self.jobs_df.iterrows()
            ]
            
            # Batch encode for better performance
            self.job_embeddings = self.sentence_model.encode(
                semantic_texts, 
                batch_size=64,  # Increased batch size
                show_progress_bar=True,
                convert_to_numpy=True
            )
            print(f"[INFO] Semantic embeddings precomputed - Shape: {self.job_embeddings.shape}")
            
        except Exception as e:
            print(f"[WARNING] Semantic embeddings setup failed: {e}")
            self.use_semantic = False

    # ------------------- Optimized Scoring -------------------
    def skill_match_vectorized(self, resume_skills_set):
        """Vectorized skill matching using cached frozensets"""
        if not resume_skills_set:
            return np.zeros(len(self.jobs_df))
        
        skill_scores = []
        for idx in range(len(self.jobs_df)):
            job_skills_set = self.skill_sets_cache[idx]
            if not job_skills_set:
                skill_scores.append(0.0)
            else:
                intersection_size = len(resume_skills_set & job_skills_set)
                skill_scores.append(intersection_size / len(job_skills_set))
        
        return np.array(skill_scores)

    def compute_scores(self, resume_skills, resume_text):
        """Optimized score computation"""
        n_jobs = len(self.jobs_df)
        
        # Preprocess resume skills once
        resume_skills_set = frozenset([s.lower().strip() for s in resume_skills if s.strip()])
        
        # Vectorized skill matching
        skill_scores = self.skill_match_vectorized(resume_skills_set)
        
        # TF-IDF similarity - optimized
        tfidf_scores = np.zeros(n_jobs)
        if self.job_vectors is not None:
            # Clean resume text once
            clean_resume_text = self._create_resume_text(resume_text, resume_skills)
            resume_vec = self.tfidf_vectorizer.transform([clean_resume_text])
            # Use efficient sparse matrix operations
            tfidf_scores = cosine_similarity(resume_vec, self.job_vectors, dense_output=False)[0].toarray().flatten()

        # Semantic similarity - optimized
        semantic_scores = np.zeros(n_jobs)
        if self.use_semantic and self.job_embeddings is not None:
            resume_emb = self.sentence_model.encode([resume_text], convert_to_numpy=True)
            # Use numpy for faster computation
            semantic_scores = np.dot(self.job_embeddings, resume_emb.T).flatten()

        # Use cached posting boosts
        posting_boosts = np.array([self.posting_boost_cache[idx] for idx in range(n_jobs)])

        # Optimized weighted combination
        combined_scores = (
            0.5 * skill_scores +
            0.3 * tfidf_scores +
            0.2 * semantic_scores
        ) * posting_boosts

        return combined_scores, skill_scores, tfidf_scores, semantic_scores

    @staticmethod
    def _create_resume_text(resume_text, resume_skills):
        """Create optimized resume text representation"""
        components = [resume_text]
        if resume_skills:
            components.append(' '.join([s.lower() for s in resume_skills]))
        
        text = ' '.join(components)
        # Simple cleaning
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return ' '.join(text.split())

    # ------------------- Optimized Job Recommendation -------------------
    def get_job_recommendations(self, resume_skills, resume_text, top_n=25):
        """Optimized job recommendation with faster processing"""
        combined_scores, skill_scores, tfidf_scores, semantic_scores = self.compute_scores(
            resume_skills, resume_text
        )

        # Pre-process resume skills for matching
        resume_skills_set = frozenset([s.lower().strip() for s in resume_skills if s.strip()])
        
        # Get top N indices efficiently
        top_indices = np.argsort(combined_scores)[::-1][:top_n]
        
        results = []
        for idx in top_indices:
            row = self.jobs_df.iloc[idx]
            
            # Calculate matched skills efficiently using cached sets
            job_skills_set = self.skill_sets_cache[idx]
            matched_skills = list(resume_skills_set & job_skills_set)
            
            # Build result dictionary
            result = {
                'company': row['company'],
                'title': row['title'],
                'position': row['position'],
                'description': row['description'],
                'skills': row['skills'],
                'remote_friendly': row['remote_friendly'],
                'location': row['location'],
                'posted': row['posted'],
                'salary_from': row['salary_from'],
                'salary_to': row['salary_to'],
                'job_url': row['job_url'],
                'match_score': round(combined_scores[idx] * 100, 2),
                'skill_match_score': round(skill_scores[idx] * 100, 2),
                'text_similarity_score': round(tfidf_scores[idx] * 100, 2),
                'semantic_similarity_score': round(semantic_scores[idx] * 100, 2),
                'matched_skills': matched_skills,
                'matched_skills_count': len(matched_skills)
            }
            
            # Generate recommendation reasons efficiently
            reasons = self._generate_reasons(result)
            result['recommendation_reasons'] = reasons
            
            results.append(result)

        return results

    @staticmethod
    def _generate_reasons(job):
        """Generate recommendation reasons efficiently"""
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
        
        return reasons

    # ------------------- Additional Utility Methods -------------------
    def get_stats(self):
        """Get matcher statistics"""
        if self.jobs_df is None:
            return None
        
        return {
            'total_jobs': len(self.jobs_df),
            'companies': self.jobs_df['company'].nunique(),
            'remote_jobs': self.jobs_df['remote_friendly'].sum(),
            'avg_skills_per_job': np.mean([len(skills) for skills in self.jobs_df['skills']]),
            'tfidf_features': self.job_vectors.shape[1] if self.job_vectors is not None else 0,
            'semantic_enabled': self.use_semantic
        }

    def clear_cache(self):
        """Clear internal caches to free memory"""
        self.skill_sets_cache.clear()
        self.posting_boost_cache.clear()
        self.job_texts_cache = None
        print("[INFO] Cache cleared")