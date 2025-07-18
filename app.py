from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, send_file
import requests
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from langdetect import detect
import csv
from io import StringIO
import os
from flask import send_file
from werkzeug.utils import secure_filename
import tempfile
import mimetypes
from bs4 import BeautifulSoup
import PyPDF2
import docx

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flash messages

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///summaries.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

API_TOKEN = os.environ.get("HF_TOKEN")
if not API_TOKEN:
    raise RuntimeError("Hugging Face API token not set. Please set the HF_TOKEN environment variable.")
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"

class Summary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_text = db.Column(db.Text, nullable=False)
    generated_summary = db.Column(db.Text, nullable=False)
    user_feedback = db.Column(db.String(20))
    improved_summary = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "mr": "Marathi",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "ru": "Russian",
    "zh": "Chinese",
    # ... aur bhi add kar sakte hain
}

def translate_text(text, src_lang, tgt_lang):
    api_url = "https://api-inference.huggingface.co/models/facebook/mbart-large-50-many-to-many-mmt"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    lang_map = {
        "en": "en_XX",
        "hi": "hi_IN",
        "bn": "bn_IN",
        "gu": "gu_IN",
        "mr": "mr_IN",
        "pa": "pa_IN",
        "ta": "ta_IN",
        "te": "te_IN",
        "ur": "ur_PK",
        "fr": "fr_XX",
        "es": "es_XX",
        "de": "de_DE",
        "ru": "ru_RU",
        "zh": "zh_CN",
        # aur bhi codes add kar sakte hain
    }
    src_code = lang_map.get(src_lang, "en_XX")
    tgt_code = lang_map.get(tgt_lang, "en_XX")
    parameters = {
        "src_lang": src_code,
        "tgt_lang": tgt_code,
    }
    payload = {"inputs": text, "parameters": parameters}
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, dict) and "translation_text" in result:
            return result["translation_text"]
        elif isinstance(result, list) and len(result) > 0 and "translation_text" in result[0]:
            return result[0]["translation_text"]
        else:
            return str(result)
    else:
        return f"Translation API Error: {response.status_code} - {response.text}"

def summarize_text(text, min_length=30, max_length=150, language="en"):
    api_url = f"https://api-inference.huggingface.co/models/{SUMMARIZATION_MODEL}"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    payload = {
        "inputs": text.strip(),  # No 'summarize:' prefix needed for bart-large-cnn
        "parameters": {
            "min_length": min_length,
            "max_length": max_length
        }
    }
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and len(result) > 0 and "summary_text" in result[0]:
            return result[0]["summary_text"]
        elif isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
            return result[0]["generated_text"]
        else:
            return str(result)
    else:
        return f"Summarization API Error: {response.status_code} - {response.text}"

def language_flag(code):
    flags = {
        'en': '🇬🇧', 'hi': '🇮🇳', 'bn': '🇧🇩', 'gu': '🇮🇳', 'mr': '🇮🇳', 'pa': '🇮🇳',
        'ta': '🇮🇳', 'te': '🇮🇳', 'ur': '🇵🇰', 'fr': '🇫🇷', 'es': '🇪🇸', 'de': '🇩🇪',
        'ru': '🇷🇺', 'zh': '🇨🇳',
        # Add more as needed
    }
    return flags.get(code, '')

def detect_language(text):
    try:
        lang = detect(text)
        if lang in LANGUAGES:
            return lang
        # If detected language is not supported, try to guess by script
    except:
        lang = None
    # Script-based fallback
    import re
    # Devanagari (Hindi, Marathi, etc.)
    if re.search(r'[\u0900-\u097F]', text):
        return 'hi'
    # Bengali
    if re.search(r'[\u0980-\u09FF]', text):
        return 'bn'
    # Gujarati
    if re.search(r'[\u0A80-\u0AFF]', text):
        return 'gu'
    # Gurmukhi (Punjabi)
    if re.search(r'[\u0A00-\u0A7F]', text):
        return 'pa'
    # Tamil
    if re.search(r'[\u0B80-\u0BFF]', text):
        return 'ta'
    # Telugu
    if re.search(r'[\u0C00-\u0C7F]', text):
        return 'te'
    # Urdu (Arabic script, but check for Urdu-specific range)
    if re.search(r'[\u0600-\u06FF]', text):
        return 'ur'
    # Default fallback
    return 'en'

def get_supported_output_languages(input_lang):
    # List of supported translation pairs (src, tgt)
    supported_pairs = {
        ("en", "hi"), ("en", "bn"), ("en", "mr"), ("en", "pa"), ("en", "ta"), ("en", "te"), ("en", "ur"), ("en", "fr"), ("en", "es"), ("en", "de"), ("en", "ru"), ("en", "zh"),
        ("hi", "en"), ("hi", "bn"), ("hi", "gu"), ("hi", "mr"), ("hi", "pa"), ("hi", "ta"), ("hi", "te"), ("hi", "ur"),
        ("bn", "en"), ("fr", "en"), ("es", "en"), ("de", "en"), ("ru", "en"), ("zh", "en"),
        # Add more supported pairs as per Helsinki-NLP opus-mt models
    }
    return {code: name for code, name in LANGUAGES.items() if (input_lang, code) in supported_pairs or input_lang == code}

def summary_to_dict(entry):
    return {
        "id": entry.id,
        "timestamp": entry.timestamp.strftime('%Y-%m-%d %H:%M'),
        "original_text": entry.original_text,
        "generated_summary": entry.generated_summary,
        "user_feedback": entry.user_feedback,
        "improved_summary": entry.improved_summary,
    }

app.jinja_env.globals.update(language_flag=language_flag)

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = None
    text = ''
    min_length = 30
    max_length = 150
    selected_language = "en"
    detected_input_language = "en"
    filtered_languages = LANGUAGES
    sorry_message = None
    # Fetch latest 10 summaries for sidebar
    recent_summaries = Summary.query.order_by(Summary.timestamp.desc()).limit(10).all()
    recent_summaries_json = [summary_to_dict(entry) for entry in recent_summaries]
    if request.method == 'POST':
        text = request.form['text']
        min_length = int(request.form.get('min_length', 30))
        max_length = int(request.form.get('max_length', 150))
        detected_input_language = detect_language(text)
        filtered_languages = get_supported_output_languages(detected_input_language)
        if not filtered_languages:
            sorry_message = f"Sorry, translation from {LANGUAGES.get(detected_input_language, detected_input_language)} is not supported yet."
        else:
            selected_language = request.form.get('language', list(filtered_languages.keys())[0])
            # Step 1: Input ko English me translate karo (agar input English nahi hai)
            if detected_input_language != "en":
                text_in_english = translate_text(text, detected_input_language, "en")
                if text_in_english.startswith("Translation API Error"):
                    sorry_message = f"Sorry, translation from {LANGUAGES.get(detected_input_language, detected_input_language)} to English is not supported yet."
                    text_in_english = None
            else:
                text_in_english = text
            # Step 2: English text se summary banao
            if text_in_english:
                summary_in_english = summarize_text(text_in_english, min_length=min_length, max_length=max_length, language="en")
                # NEW LOGIC: If input is shorter than summary, return input (or translated input)
                if len(text.strip()) < len(summary_in_english.strip()):
                    if detected_input_language == selected_language:
                        summary = text
                    else:
                        summary = translate_text(text, detected_input_language, selected_language)
                else:
                    # Step 3: Agar output language English nahi hai, toh summary ko output lang me translate karo
                    if selected_language != "en":
                        summary_translated = translate_text(summary_in_english, "en", selected_language)
                        if summary_translated.startswith("Translation API Error"):
                            sorry_message = f"Sorry, translation from English to {LANGUAGES.get(selected_language, selected_language)} is not supported yet."
                            summary = None
                        else:
                            summary = summary_translated
                    else:
                        summary = summary_in_english
    else:
        filtered_languages = get_supported_output_languages(detected_input_language)
        if not filtered_languages:
            sorry_message = f"Sorry, translation from {LANGUAGES.get(detected_input_language, detected_input_language)} is not supported yet."
    return render_template('index.html', summary=summary, text=text, languages=filtered_languages, selected_language=selected_language, detected_input_language=detected_input_language, sorry_message=sorry_message, recent_summaries=recent_summaries, recent_summaries_json=recent_summaries_json)

@app.route('/api/summarize', methods=['POST'])
def api_summarize():
    data = request.get_json()
    text = data.get('text', '')
    min_length = int(data.get('min_length', 30))
    max_length = int(data.get('max_length', 150))
    language = data.get('language', 'en')
    input_language = data.get('input_language', 'en')
    if input_language != language:
        translated_text = translate_text(text, input_language, language)
        summary = summarize_text(translated_text, min_length=min_length, max_length=max_length, language=language)
    else:
        summary = summarize_text(text, min_length=min_length, max_length=max_length, language=language)
    return jsonify({'summary': summary})

@app.route('/feedback', methods=['POST'])
def feedback():
    original_text = request.form.get('original_text', '')
    generated_summary = request.form.get('generated_summary', '')
    user_feedback = request.form.get('user_feedback', '')
    improved_summary = request.form.get('improved_summary', '')
    if original_text and generated_summary:
        entry = Summary(
            original_text=original_text,
            generated_summary=generated_summary,
            user_feedback=user_feedback,
            improved_summary=improved_summary
        )
        db.session.add(entry)
        db.session.commit()
        flash('Feedback saved successfully!', 'success')
    else:
        flash('Feedback not saved. Missing data.', 'danger')
    return redirect(url_for('index'))

@app.route('/history')
def history():
    all_summaries = Summary.query.order_by(Summary.timestamp.desc()).all()
    summaries_dict = [summary_to_dict(entry) for entry in all_summaries]
    return render_template('history.html', summaries=all_summaries, summaries_json=summaries_dict)

@app.route('/export')
def export():
    all_summaries = Summary.query.order_by(Summary.timestamp.desc()).all()
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(['original_text', 'generated_summary', 'user_feedback', 'improved_summary', 'timestamp'])
    for entry in all_summaries:
        cw.writerow([
            entry.original_text,
            entry.generated_summary,
            entry.user_feedback,
            entry.improved_summary,
            entry.timestamp.strftime('%Y-%m-%d %H:%M')
        ])
    output = si.getvalue()
    return app.response_class(
        output,
        mimetype='text/csv',
        headers={"Content-Disposition": "attachment;filename=summaries_feedback.csv"}
    )

@app.route('/extract_url_text', methods=['POST'])
def extract_url_text():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({'error': 'No URL provided.'}), 400
    try:
        import requests
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        # Try to get main content
        main = soup.find('main')
        if main:
            text = main.get_text(separator=' ', strip=True)
        else:
            # Fallback: get all paragraphs
            paragraphs = soup.find_all('p')
            text = '\n'.join([p.get_text(separator=' ', strip=True) for p in paragraphs])
        if not text.strip():
            text = soup.get_text(separator=' ', strip=True)
        return jsonify({'text': text.strip()})
    except Exception as e:
        return jsonify({'error': f'Failed to extract text: {str(e)}'}), 500

@app.route('/extract_file_text', methods=['POST'])
def extract_file_text():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
    file = request.files['file']
    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[-1].lower()
    try:
        if ext == 'pdf':
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() or ''
        elif ext == 'docx':
            doc = docx.Document(file)
            text = '\n'.join([para.text for para in doc.paragraphs])
        elif ext == 'txt':
            text = file.read().decode('utf-8', errors='ignore')
        else:
            return jsonify({'error': 'Unsupported file type.'}), 400
        return jsonify({'text': text.strip()})
    except Exception as e:
        return jsonify({'error': f'Failed to extract text: {str(e)}'}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
   
   
    app.run(debug=True) 