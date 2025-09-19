import re
import json
from collections import defaultdict, Counter
import string

class SkillExtractor:
    def __init__(self, skills_db_path=None):
        # Comprehensive skill database - you can expand this
        self.skill_keywords = {
            # Programming Languages
            'programming': {
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
                'ruby', 'php', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl',
                'shell', 'bash', 'powershell', 'sql', 'html', 'css', 'dart', 'lua'
            },
            
            # Frameworks and Libraries
            'frameworks': {
                'react', 'angular', 'vue', 'django', 'flask', 'fastapi', 'express',
                'spring', 'laravel', 'rails', 'asp.net', 'nodejs', 'nextjs', 'nuxt',
                'bootstrap', 'tailwind', 'jquery', 'tensorflow', 'pytorch', 'scikit-learn',
                'pandas', 'numpy', 'matplotlib', 'seaborn', 'opencv'
            },
            
            # Databases
            'databases': {
                'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'sqlite',
                'oracle', 'dynamodb', 'cassandra', 'neo4j', 'influxdb', 'mariadb'
            },
            
            # Cloud and DevOps
            'cloud': {
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible',
                'jenkins', 'gitlab', 'github', 'circleci', 'travis', 'heroku', 'vercel'
            },
            
            # Tools and Software
            'tools': {
                'git', 'jira', 'confluence', 'slack', 'trello', 'asana', 'figma',
                'sketch', 'photoshop', 'illustrator', 'excel', 'powerpoint', 'tableau',
                'power bi', 'looker', 'grafana', 'postman', 'insomnia'
            },
            
            # Methodologies
            'methodologies': {
                'agile', 'scrum', 'kanban', 'devops', 'ci/cd', 'tdd', 'bdd',
                'microservices', 'rest', 'graphql', 'soap', 'api', 'mvc'
            },
            
            # Soft Skills
            'soft_skills': {
                'leadership', 'communication', 'teamwork', 'problem solving',
                'critical thinking', 'time management', 'project management',
                'analytical', 'creative', 'adaptability', 'collaboration'
            }
        }
        
        # Flatten all skills for quick lookup
        self.all_skills = set()
        for category_skills in self.skill_keywords.values():
            self.all_skills.update(category_skills)
        
        # Create skill variations (plurals, common variations)
        self._expand_skill_variations()
        
        # Load additional skills from file if provided
        if skills_db_path:
            self._load_skills_db(skills_db_path)
        
        # Pre-compile regex patterns for performance
        self._compile_patterns()
        
    def _expand_skill_variations(self):
        """Expand skill database with common variations"""
        expanded_skills = set(self.all_skills)
        
        for skill in list(self.all_skills):
            # Add plural forms
            if not skill.endswith('s') and skill not in ['css', 'express']:
                expanded_skills.add(skill + 's')
            
            # Add common variations
            variations = {
                'javascript': ['js', 'javascript'],
                'typescript': ['ts', 'typescript'],
                'python': ['py', 'python'],
                'react': ['reactjs', 'react.js'],
                'node': ['nodejs', 'node.js'],
                'c++': ['cpp', 'c plus plus'],
                'c#': ['csharp', 'c sharp'],
                'postgresql': ['postgres', 'psql'],
                'mongodb': ['mongo'],
                'machine learning': ['ml', 'machine learning'],
                'artificial intelligence': ['ai', 'artificial intelligence']
            }
            
            if skill in variations:
                expanded_skills.update(variations[skill])
        
        self.all_skills = expanded_skills
    
    def _load_skills_db(self, skills_db_path):
        """Load additional skills from JSON file"""
        try:
            with open(skills_db_path, 'r') as f:
                additional_skills = json.load(f)
                if isinstance(additional_skills, list):
                    self.all_skills.update([s.lower() for s in additional_skills])
                elif isinstance(additional_skills, dict):
                    for category, skills in additional_skills.items():
                        if category not in self.skill_keywords:
                            self.skill_keywords[category] = set()
                        self.skill_keywords[category].update([s.lower() for s in skills])
                        self.all_skills.update([s.lower() for s in skills])
        except Exception as e:
            print(f"[WARNING] Could not load skills database: {e}")
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance"""
        # Pattern for extracting skill-like phrases
        self.skill_pattern = re.compile(
            r'\b(?:' + '|'.join(re.escape(skill) for skill in self.all_skills) + r')\b',
            re.IGNORECASE
        )
        
        # Pattern for common skill contexts
        self.context_patterns = {
            'experience': re.compile(r'(?:experience (?:in|with)|skilled (?:in|with)|proficient (?:in|with)|expert (?:in|with))[\s:]+([\w\s,.-]+)', re.IGNORECASE),
            'technologies': re.compile(r'(?:technologies|tools|frameworks|languages)[\s:]+([\w\s,.-]+)', re.IGNORECASE),
            'skills': re.compile(r'(?:skills|competencies|abilities)[\s:]+([\w\s,.-]+)', re.IGNORECASE)
        }
    
    def clean_text(self, text):
        """Clean and normalize text for better extraction"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        
        # Remove bullet points and special characters but keep important ones
        text = re.sub(r'[•▪▫◦‣⁃]', ' ', text)
        text = re.sub(r'[^\w\s.#+-]', ' ', text)  # Keep dots, hashes, plus, minus
        
        return text.lower()
    
    def extract_skills_direct(self, text):
        """Direct skill matching using pre-compiled patterns"""
        if not text:
            return []
        
        cleaned_text = self.clean_text(text)
        matches = self.skill_pattern.findall(cleaned_text)
        return list(set(matches))
    
    def extract_skills_contextual(self, text):
        """Extract skills from contextual patterns"""
        skills = []
        
        for pattern_name, pattern in self.context_patterns.items():
            matches = pattern.findall(text)
            for match in matches:
                # Split by common separators and extract individual skills
                potential_skills = re.split(r'[,;|\n\t]', match)
                for skill in potential_skills:
                    skill = skill.strip()
                    if skill and len(skill) > 1:
                        # Check if it's a known skill
                        skill_lower = skill.lower()
                        if skill_lower in self.all_skills:
                            skills.append(skill_lower)
        
        return list(set(skills))
    
    def extract_from_sections(self, text):
        """Extract skills from specific resume sections"""
        skills = []
        
        # Define section patterns
        section_patterns = {
            'technical_skills': re.compile(r'(?:technical skills|technical competencies|programming languages|technologies):(.*?)(?=\n[A-Z]|\n\n|$)', re.IGNORECASE | re.DOTALL),
            'skills': re.compile(r'(?:^|\n)skills:(.*?)(?=\n[A-Z]|\n\n|$)', re.IGNORECASE | re.DOTALL),
            'tools': re.compile(r'(?:tools & technologies|tools and technologies|software):(.*?)(?=\n[A-Z]|\n\n|$)', re.IGNORECASE | re.DOTALL)
        }
        
        for section_name, pattern in section_patterns.items():
            matches = pattern.findall(text)
            for match in matches:
                section_skills = self.extract_skills_direct(match)
                skills.extend(section_skills)
        
        return list(set(skills))
    
    def extract_skills(self, text, include_soft_skills=True, confidence_threshold=0.0):
        """
        Main method to extract skills from text
        
        Args:
            text (str): Input text (resume, job description, etc.)
            include_soft_skills (bool): Whether to include soft skills
            confidence_threshold (float): Minimum confidence score (0-1)
        
        Returns:
            dict: Extracted skills with metadata
        """
        if not text:
            return {'skills': [], 'skill_categories': {}, 'confidence_scores': {}}
        
        all_extracted_skills = []
        
        # Method 1: Direct pattern matching
        direct_skills = self.extract_skills_direct(text)
        all_extracted_skills.extend(direct_skills)
        
        # Method 2: Contextual extraction
        contextual_skills = self.extract_skills_contextual(text)
        all_extracted_skills.extend(contextual_skills)
        
        # Method 3: Section-based extraction
        section_skills = self.extract_from_sections(text)
        all_extracted_skills.extend(section_skills)
        
        # Count occurrences for confidence scoring
        skill_counts = Counter(all_extracted_skills)
        
        # Filter by confidence threshold and soft skills preference
        final_skills = []
        skill_categories = defaultdict(list)
        confidence_scores = {}
        
        for skill, count in skill_counts.items():
            # Calculate simple confidence based on occurrence and length
            confidence = min(1.0, (count * 0.3) + (len(skill) * 0.1) / 10)
            
            if confidence >= confidence_threshold:
                # Check if it's a soft skill
                is_soft_skill = skill in self.skill_keywords.get('soft_skills', set())
                
                if include_soft_skills or not is_soft_skill:
                    final_skills.append(skill)
                    confidence_scores[skill] = confidence
                    
                    # Categorize the skill
                    for category, category_skills in self.skill_keywords.items():
                        if skill in category_skills:
                            skill_categories[category].append(skill)
                            break
                    else:
                        skill_categories['other'].append(skill)
        
        # Remove duplicates and sort
        final_skills = sorted(list(set(final_skills)))
        
        return {
            'skills': final_skills,
            'skill_categories': dict(skill_categories),
            'confidence_scores': confidence_scores,
            'total_found': len(final_skills)
        }
    
    def get_skill_suggestions(self, text, max_suggestions=10):
        """Get skill suggestions based on partial matches"""
        if not text:
            return []
        
        text_lower = text.lower()
        suggestions = []
        
        for skill in self.all_skills:
            if text_lower in skill or skill in text_lower:
                suggestions.append(skill)
        
        return sorted(suggestions)[:max_suggestions]
    
    def validate_skills(self, skills):
        """Validate if provided skills are recognized"""
        if not skills:
            return []
        
        valid_skills = []
        for skill in skills:
            if isinstance(skill, str) and skill.lower().strip() in self.all_skills:
                valid_skills.append(skill.lower().strip())
        
        return valid_skills
    
    def get_stats(self):
        """Get extractor statistics"""
        return {
            'total_skills_in_db': len(self.all_skills),
            'categories': list(self.skill_keywords.keys()),
            'skills_per_category': {cat: len(skills) for cat, skills in self.skill_keywords.items()}
        }