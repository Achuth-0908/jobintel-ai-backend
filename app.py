from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import PyPDF2
import json
import traceback
from datetime import datetime

# Import your ML models
try:
    from ml_models.skill_extractor import SkillExtractor
    from ml_models.job_matcher import JobMatcher
except ImportError as e:
    print(f"Warning: Could not import ML models: {e}")
    SkillExtractor = None
    JobMatcher = None

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize models with error handling
skill_extractor = None
job_matcher = None

try:
    if SkillExtractor:
        skill_extractor = SkillExtractor()
        print("✓ SkillExtractor initialized successfully")
    if JobMatcher:
        job_matcher = JobMatcher()
        print("✓ JobMatcher initialized successfully")
except Exception as e:
    print(f"⚠️  Error initializing models: {e}")
    traceback.print_exc()

ALLOWED_EXTENSIONS = {'pdf', 'txt'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file with better error handling"""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Check if PDF is encrypted
            if pdf_reader.is_encrypted:
                return "Error: PDF is encrypted and cannot be processed"
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as page_error:
                    print(f"Error extracting text from page {page_num}: {page_error}")
                    continue
        
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return f"Error reading PDF: {str(e)}"

def extract_text_from_txt(txt_path):
    """Extract text from TXT file with encoding handling"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(txt_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error reading file with {encoding}: {e}")
            continue
    
    return f"Error: Could not read file with any supported encoding"

@app.route('/')
def index():
    """API information endpoint"""
    return jsonify({
        'message': 'JobIntel AI API',
        'status': 'running',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'endpoints': {
            'upload': '/upload (POST) - Upload resume file for analysis',
            'analyze_text': '/analyze_text (POST) - Analyze resume text directly',
            'health': '/health (GET) - Health check'
        },
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size': '16MB'
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis"""
    try:
        # Check if models are available
        if not skill_extractor or not job_matcher:
            return jsonify({
                'error': 'ML models not initialized',
                'details': 'SkillExtractor or JobMatcher failed to load'
            }), 500

        # Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file or not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type', 
                'supported_formats': list(ALLOWED_EXTENSIONS)
            }), 400

        # Save and process file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Extract text based on file type
            if filename.lower().endswith('.pdf'):
                resume_text = extract_text_from_pdf(file_path)
            else:
                resume_text = extract_text_from_txt(file_path)

            # Clean up file immediately
            if os.path.exists(file_path):
                os.remove(file_path)

            # Validate extracted text
            if not resume_text or not resume_text.strip():
                return jsonify({'error': 'Could not extract text from file'}), 400

            if resume_text.startswith('Error'):
                return jsonify({'error': resume_text}), 400

            # Analyze skills
            print("Extracting skills...")
            skill_analysis = skill_extractor.extract_skills(resume_text)
            print(f"Skills extracted: {len(skill_analysis.get('skills', []))}")

            # Get job recommendations
            print("Getting job recommendations...")
            job_recommendations = job_matcher.get_job_recommendations(
                skill_analysis.get('skills', []),  # Use .get() to avoid KeyError
                resume_text,
                top_n=25
            )
            print(f"Job recommendations: {len(job_recommendations)}")

            # Prepare response
            result = {
                'success': True,
                'filename': filename,
                'resume_text': resume_text[:1000] + '...' if len(resume_text) > 1000 else resume_text,
                'skills': skill_analysis,
                'job_recommendations': job_recommendations,
                'stats': {
                    'text_length': len(resume_text),
                    'skills_found': len(skill_analysis.get('skills', [])),
                    'recommendations_count': len(job_recommendations)
                }
            }

            return jsonify(result)

        except Exception as processing_error:
            # Clean up file on error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise processing_error

    except Exception as e:
        print(f"Error in upload_file: {e}")
        traceback.print_exc()
        return jsonify({
            'error': 'An error occurred while processing your file',
            'details': str(e) if app.debug else None
        }), 500

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    """Analyze resume text directly without file upload"""
    try:
        # Check if models are available
        if not skill_extractor or not job_matcher:
            return jsonify({
                'error': 'ML models not initialized',
                'details': 'SkillExtractor or JobMatcher failed to load'
            }), 500

        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400

        data = request.get_json(silent=True)
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        resume_text = data.get('text', '').strip()
        if not resume_text:
            return jsonify({'error': 'Please provide resume text'}), 400

        # Validate text length
        if len(resume_text) < 50:
            return jsonify({'error': 'Resume text is too short. Please provide more detailed text.'}), 400

        # Analyze skills
        print("Extracting skills from text...")
        skill_analysis = skill_extractor.extract_skills(resume_text)
        print(f"Skills extracted: {len(skill_analysis.get('skills', []))}")

        # Get job recommendations
        print("Getting job recommendations...")
        job_recommendations = job_matcher.get_job_recommendations(
            skill_analysis.get('skills', []),  # Use .get() to avoid KeyError
            resume_text,
            top_n=25
        )
        print(f"Job recommendations: {len(job_recommendations)}")

        # Prepare response
        result = {
            'success': True,
            'resume_text': resume_text[:1000] + '...' if len(resume_text) > 1000 else resume_text,
            'skills': skill_analysis,
            'job_recommendations': job_recommendations,
            'stats': {
                'text_length': len(resume_text),
                'skills_found': len(skill_analysis.get('skills', [])),
                'recommendations_count': len(job_recommendations)
            }
        }

        return jsonify(result)

    except Exception as e:
        print(f"Error in analyze_text: {e}")
        traceback.print_exc()
        return jsonify({
            'error': 'An error occurred while analyzing the text',
            'details': str(e) if app.debug else None
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        # Check model status
        models_status = {
            'skill_extractor_loaded': skill_extractor is not None,
            'job_matcher_loaded': job_matcher is not None,
        }

        # Additional checks if models are loaded
        if skill_extractor:
            try:
                # Test skill extraction with simple text
                test_result = skill_extractor.extract_skills("Python programming experience")
                models_status['skill_extractor_working'] = len(test_result.get('skills', [])) > 0
            except Exception as e:
                models_status['skill_extractor_working'] = False
                models_status['skill_extractor_error'] = str(e)

        if job_matcher:
            try:
                # Check if job matcher has data
                stats = job_matcher.get_stats() if hasattr(job_matcher, 'get_stats') else None
                models_status['job_matcher_working'] = stats is not None
                if stats:
                    models_status['job_data'] = stats
            except Exception as e:
                models_status['job_matcher_working'] = False
                models_status['job_matcher_error'] = str(e)

        # Overall health status
        is_healthy = (
            models_status.get('skill_extractor_loaded', False) and 
            models_status.get('job_matcher_loaded', False) and
            models_status.get('skill_extractor_working', False) and
            models_status.get('job_matcher_working', False)
        )

        return jsonify({
            'status': 'healthy' if is_healthy else 'degraded',
            'message': 'JobIntel AI is running' if is_healthy else 'Some components may not be working properly',
            'timestamp': datetime.now().isoformat(),
            'models': models_status,
            'system': {
                'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
                'max_content_length': app.config['MAX_CONTENT_LENGTH']
            }
        }), 200 if is_healthy else 503

    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'message': 'Health check failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'error': 'File too large',
        'message': 'File size exceeds the maximum limit of 16MB'
    }), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    
    print(f"🚀 Starting JobIntel AI API on port {port}...")
    print(f"📊 Debug mode: {debug}")
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"💾 Max file size: {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)}MB")
    
    if skill_extractor and job_matcher:
        print("✅ All models loaded successfully")
    else:
        print("⚠️  Some models failed to load - check logs above")
    
    app.run(debug=debug, host='0.0.0.0', port=port)