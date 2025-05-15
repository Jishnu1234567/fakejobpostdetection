import os
import docx2txt
import requests
from bs4 import BeautifulSoup
from django.shortcuts import render
from .forms import JobPostForm
from .ml_model import load_model_and_vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from pdfminer.high_level import extract_text as extract_pdf_text
from io import BytesIO

model, vectorizer = load_model_and_vectorizer()

def extract_text_from_file(file):
    filename = file.name.lower()
    if filename.endswith('.pdf'):
        return extract_pdf_text(BytesIO(file.read()))
    elif filename.endswith('.docx'):
        return docx2txt.process(file)
    elif filename.endswith('.txt'):
        return file.read().decode('utf-8')
    else:
        return None

def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        texts = [p.get_text() for p in soup.find_all('p')]
        return "\n".join(texts) if texts else None
    except Exception:
        return None

def home(request):
    prediction = None
    confidence = None
    error = None

    if request.method == 'POST':
        form = JobPostForm(request.POST, request.FILES)

        if form.is_valid():
            input_text = ""

            # 1. Manual text input
            if form.cleaned_data.get('job_text'):
                input_text = form.cleaned_data['job_text']

            # 2. File upload
            elif form.cleaned_data.get('job_file'):
                extracted = extract_text_from_file(form.cleaned_data['job_file'])
                if extracted:
                    input_text = extracted
                else:
                    error = "Unsupported or unreadable file format."

            # 3. Job URL
            elif form.cleaned_data.get('job_url'):
                extracted = extract_text_from_url(form.cleaned_data['job_url'])
                if extracted:
                    input_text = extracted
                else:
                    error = "Error extracting text from the URL."

            if input_text:
                X = vectorizer.transform([input_text])
                result = model.predict(X)[0]
                proba = model.predict_proba(X).max()
                prediction = "Fake" if result == 1 else "Real"
                confidence = round(proba * 100, 2)
            elif not error:
                error = "Please provide job text, file, or URL."

        else:
            error = "Invalid form submission."

    else:
        form = JobPostForm()

    return render(request, 'home.html', {
        'form': form,
        'prediction': prediction,
        'confidence': confidence,
        'error': error,
    })
