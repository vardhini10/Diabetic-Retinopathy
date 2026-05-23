from flask import Flask, request, render_template, send_file, redirect, url_for
from PIL import Image
import numpy as np
import base64
import io
import os

from image_validator import detect_eye
from models import EyeSeverityDetector

app = Flask(__name__)
app.config['SECRET_KEY'] = 'replace-with-secure-key'  # required for session cookies
app.config['last_analysis'] = None


def generate_report_pdf(image: Image.Image, severity: str, confidence: float, patient_name: str, patient_age: int):
    """Generate a PDF report (A4) of the analysis."""
    from fpdf import FPDF
    from datetime import datetime

    # Map internal severity to report-friendly label
    severity_label_map = {
        'normal': 'Normal',
        'mild': 'Mild',
        'moderate': 'Moderate',
        'severe': 'Severe',
        'proliferative': 'Proliferative DR'
    }
    severity_display = severity_label_map.get(severity.lower(), severity.title())

    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font('Arial', 'B', 18)
    pdf.cell(0, 12, 'DIABETIC RETINOPATHY ANALYSIS REPORT', ln=True, align='C')
    pdf.ln(4)

    # Patient details line
    report_date = datetime.now().strftime('%Y-%m-%d')
    patient_line = (
        f"Patient Name: {patient_name or '________'}    "
        f"Age: {patient_age if patient_age not in (None, 0, '') else '____'}    "
        f"Date: {report_date}"
    )
    pdf.set_font('Arial', size=12)
    pdf.cell(0, 8, patient_line, ln=True, align='L')
    pdf.ln(6)

    # Image
    try:
        temp_path = os.path.join('models', 'temp_eye_image.jpg')
        image.save(temp_path)
        pdf.image(temp_path, x=15, w=180)
        pdf.ln(95)
    except Exception:
        pdf.ln(18)

    # Analysis and recommendations
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Analysis Result', ln=True)
    pdf.set_font('Arial', size=12)
    pdf.cell(0, 8, f'Severity Level: {severity_display}', ln=True)
    pdf.cell(0, 8, f'Confidence: {confidence * 100:.2f}%', ln=True)
    pdf.ln(4)

    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Recommendations', ln=True)
    pdf.set_font('Arial', size=12)
    recs = [
        'Schedule an eye examination with an ophthalmologist',
        'Maintain strict blood glucose control',
        'Monitor blood pressure regularly',
        'Follow up for eye screening in 6 months'
    ]
    for rec in recs:
        pdf.multi_cell(0, 7, f'- {rec}')
    pdf.ln(6)

    pdf.set_font('Arial', size=10)
    advisory = (
        'This AI-generated report is for screening purposes only. '
        'Please consult a qualified ophthalmologist for proper diagnosis and treatment.'
    )
    pdf.multi_cell(0, 6, advisory)

    return pdf.output(dest='S').encode('latin1')


# simple homepage with upload form
@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    severity = None
    confidence = None
    image_data = None

    if request.method == 'POST':
        file = request.files.get('image')
        patient_name = request.form.get('patient_name', '')
        patient_age = request.form.get('patient_age', '')

        if not file:
            error = 'No file selected.'
            return render_template('index.html', error=error)

        try:
            # read image bytes and keep for display
            img_bytes = file.read()
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            image_array = np.array(image)
        except Exception as e:
            error = f'Unable to read image: {e}'
            return render_template('index.html', error=error)

        # validation: ensure an eye is present
        if not detect_eye(image_array):
            error = 'This image is not supported. Please upload a clear eye image.'
            return render_template('index.html', error=error)

        # Run AI analysis - independent of registration data
        detector = EyeSeverityDetector()
        result = detector.classify_severity(image)
        raw_sev = result.get('severity', 'unknown')
        # map to display labels
        label_map = {
            'normal': 'Normal',
            'mild': 'Mild',
            'moderate': 'Moderate',
            'severe': 'Severe',
            'proliferative': 'Proliferative DR'
        }
        severity = label_map.get(raw_sev.lower(), raw_sev)
        confidence = result.get('confidence', 0.0)

        # record last analysis for download
        app.config['last_analysis'] = {
            'image': image,
            'severity': raw_sev,
            'confidence': confidence,
            'patient_name': patient_name,
            'patient_age': patient_age
        }

        # prepare image for embedding (base64)
        encoded = base64.b64encode(img_bytes).decode('utf-8')
        image_data = f'data:image/jpeg;base64,{encoded}'

    return render_template('index.html', error=error,
                           severity=severity,
                           confidence=confidence,
                           image_data=image_data)


@app.route('/download_report')
def download_report():
    analysis = app.config.get('last_analysis')
    if not analysis:
        return redirect(url_for('index'))

    pdf_bytes = generate_report_pdf(
        image=analysis['image'],
        severity=analysis['severity'],
        confidence=analysis['confidence'],
        patient_name=analysis.get('patient_name', ''),
        patient_age=int(analysis.get('patient_age') or 0)
    )

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        download_name='analysis_report.pdf',
        as_attachment=True
    )


if __name__ == '__main__':
    app.run(debug=True)
