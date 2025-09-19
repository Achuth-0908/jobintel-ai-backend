import re
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import numpy as np
import logging

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class SkillExtractor:
    def __init__(self, semantic_threshold=0.35):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("SkillExtractor")

        # Load spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Load semantic model if available
        self.sentence_model = None
        self.semantic_threshold = semantic_threshold
        if SentenceTransformer is not None:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                self.logger.warning(f"Could not load sentence transformer: {e}")
                self.sentence_model = None

        # Skills database
        self.skills_database = self._load_skills_database()
        self.all_skills = sorted(set(skill for skills in self.skills_database.values() for skill in skills))

        # Precompute embeddings for semantic similarity
        if self.sentence_model:
            try:
                self.skill_embeddings = self.sentence_model.encode(self.all_skills, show_progress_bar=False)
            except Exception as e:
                self.logger.warning(f"Error computing skill embeddings: {e}")
                self.skill_embeddings = None
                self.sentence_model = None

    def _load_skills_database(self):
        """Define categories and skills"""
        return {
            'programming': [
                'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
                'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql', 'html', 'css', 'react',
                'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring',
                'laravel', 'rails', 'asp.net', 'jquery', 'bootstrap', 'typescript', 'nextjs',
                'react native', 'flutter', 'xamarin', 'ionic', 'cordova', 'electron'
            ],
            'data_science': [
                'machine learning', 'deep learning', 'data analysis', 'statistics',
                'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras',
                'matplotlib', 'seaborn', 'plotly', 'tableau', 'power bi', 'excel',
                'nlp', 'computer vision', 'big data', 'hadoop', 'spark', 'data mining'
            ],
            'cloud_devops': [
                'aws', 'azure', 'google cloud', 'docker', 'kubernetes', 'jenkins',
                'git', 'github', 'gitlab', 'terraform', 'ansible', 'ci/cd', 'linux',
                'bash', 'shell scripting', 'monitoring', 'logging', 'prometheus', 'grafana'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'oracle',
                'sqlite', 'dynamodb', 'elasticsearch', 'neo4j', 'firebase'
            ],
            'tools': [
                'jira', 'confluence', 'slack', 'trello', 'asana', 'notion',
                'postman', 'swagger', 'figma', 'photoshop', 'illustrator'
            ],
            'soft_skills': [
                'leadership', 'communication', 'teamwork', 'problem solving',
                'project management', 'agile', 'scrum', 'creativity', 'adaptability'
            ]
        }

    def preprocess_text(self, text):
        """Normalize text for skill extraction"""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s+.#\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # ---------------- Regex Extraction ----------------
    def extract_skills_regex(self, text):
        text = self.preprocess_text(text)
        found_skills = [skill for skill in self.all_skills if re.search(r'\b' + re.escape(skill) + r'\b', text)]
        return list(set(found_skills))

    # ---------------- spaCy NLP Extraction ----------------
    def extract_skills_spacy(self, text):
        if not self.nlp:
            return []
        try:
            doc = self.nlp(text)
            found_skills = []
            phrases = [chunk.text.lower() for chunk in doc.noun_chunks] + [ent.text.lower() for ent in doc.ents]
            for skill in self.all_skills:
                if any(skill in phrase for phrase in phrases):
                    found_skills.append(skill)
            return list(set(found_skills))
        except Exception as e:
            self.logger.warning(f"spaCy extraction error: {e}")
            return []

    # ---------------- Semantic Extraction ----------------
    def extract_skills_semantic(self, text):
        if not self.sentence_model or self.skill_embeddings is None:
            return []
        try:
            text_embedding = self.sentence_model.encode([self.preprocess_text(text)])
            similarities = cosine_similarity(text_embedding, self.skill_embeddings)[0]
            return [self.all_skills[i] for i, sim in enumerate(similarities) if sim >= self.semantic_threshold]
        except Exception as e:
            self.logger.warning(f"Semantic extraction error: {e}")
            return []

    # ---------------- Main Extraction ----------------
    def extract_skills(self, text):
        regex_skills = self.extract_skills_regex(text)
        spacy_skills = self.extract_skills_spacy(text)
        semantic_skills = self.extract_skills_semantic(text)
        all_skills = list(set(regex_skills + spacy_skills + semantic_skills))

        categorized_skills = {cat: [s for s in all_skills if s in skills] for cat, skills in self.skills_database.items()}
        return {
            'all_skills': all_skills,
            'categorized_skills': categorized_skills,
            'skill_count': len(all_skills)
        }

    # ---------------- Frequency ----------------
    def get_skill_frequency(self, text):
        text = self.preprocess_text(text)
        freq = {skill: len(re.findall(r'\b' + re.escape(skill) + r'\b', text)) for skill in self.all_skills}
        return {k: v for k, v in freq.items() if v > 0}
