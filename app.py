from flask import Flask, request # type: ignore
from flask_cors import CORS # type: ignore
from werkzeug.utils import secure_filename # type: ignore
import os
import PyPDF2 # type: ignore
import json
from ml_models.skill_extractor import SkillExtractor
from ml_models.job_matcher import JobMatcher
import traceback

app = Flask(__name__)
CORS(app)  # <-- enables CORS

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

skill_extractor = SkillExtractor()
job_matcher = JobMatcher()

ALLOWED_EXTENSIONS = {'pdf', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_txt(txt_path):
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error extracting text from TXT: {e}")
        return ""

@app.route('/')
def index():
    return {
        'message': 'JobIntel AI API',
        'status': 'running',
        'endpoints': {
            'upload': '/upload (POST)',
            'analyze_text': '/analyze_text (POST)',
            'health': '/health (GET)'
        }
    }

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return json_response({'error': 'No file uploaded'}, 400)

        file = request.files['file']
        if file.filename == '':
            return json_response({'error': 'No file selected'}, 400)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            resume_text = extract_text_from_pdf(file_path) if filename.lower().endswith('.pdf') else extract_text_from_txt(file_path)

            if not resume_text.strip():
                return json_response({'error': 'Could not extract text from file'}, 400)

            skill_analysis = skill_extractor.extract_skills(resume_text)
            job_recommendations = job_matcher.get_job_recommendations(
                skill_analysis['all_skills'],
                resume_text
            )

            os.remove(file_path)

            result = {
                'success': True,
                'resume_text': resume_text[:500] + '...' if len(resume_text) > 500 else resume_text,
                'skills': skill_analysis,
                'job_recommendations': json_safe(job_recommendations)
            }

            print("Result:", json.dumps(result, indent=2))
            return json_response(result)

        return json_response({'error': 'Invalid file type. Please upload PDF or TXT files.'}, 400)

    except Exception as e:
        print(f"Error in upload_file: {e}")
        traceback.print_exc()
        return json_response({'error': 'An error occurred while processing your file'}, 500)

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    try:
        if not request.is_json:
            return json_response({'error': 'Content-Type must be application/json'}, 400)
        data = request.get_json(silent=True)
        if not data:
            return json_response({'error': 'No JSON data provided'}, 400)

        resume_text = data.get('text', '')
        if not resume_text.strip():
            return json_response({'error': 'Please provide resume text'}, 400)

        skill_analysis = skill_extractor.extract_skills(resume_text)
        job_recommendations = job_matcher.get_job_recommendations(
            skill_analysis['all_skills'],
            resume_text
        )

        result = {
            'success': True,
            'resume_text': resume_text[:500] + '...' if len(resume_text) > 500 else resume_text,
            'skills': skill_analysis,
            'job_recommendations': json_safe(job_recommendations)
        }

        print("Result:", json.dumps(result, indent=2))
        return json_response(result)

    except Exception as e:
        print(f"Error in analyze_text: {e}")
        traceback.print_exc()
        return json_response({'error': 'An error occurred while analyzing the text'}, 500)

@app.route('/health')
def health_check():
    return json_response({
        'status': 'healthy',
        'message': 'JobIntel AI is running',
        'models_loaded': {
            'skill_extractor': skill_extractor is not None,
            'job_matcher': job_matcher is not None,
            'spacy_model': skill_extractor.nlp is not None if skill_extractor else False,
            'sentence_transformer': skill_extractor.sentence_model is not None if skill_extractor else False
        }
    })

def json_safe(obj):
    if isinstance(obj, list):
        return [json_safe(o) for o in obj]
    elif isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, float):
        return round(float(obj), 4)
    elif hasattr(obj, 'item'):
        return obj.item()
    return obj

def json_response(data, status=200):
    return app.response_class(
        response=json.dumps(json_safe(data), indent=2),
        status=status,
        mimetype='application/json'
    )

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting JobIntel AI API on port {port}...")
    app.run(debug=False, host='0.0.0.0', port=port)
