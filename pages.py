# Page components for the application

import streamlit as st
import cv2
from PIL import Image
import pandas as pd
import os
from datetime import datetime
import numpy as np
from auth import logout_user, get_current_user, load_users
from models import EyeSeverityDetector
from utils import save_prediction, get_user_predictions, format_severity_display, get_statistics, get_severity_color, normalize_severity_label
from config import UPLOAD_DIR
from config import SEVERITY_LEVELS
from image_validator import validate_uploaded_image, get_validation_details, is_retinal_image, classify_external_eye_severity
from state import init_state


def set_app_background(opacity=0.25):
    """Set eye-diagram background for all pages."""
    st.markdown(f"""
    <style>
    .stApp {{
        background: url('models/eye_diagram.png') center/cover no-repeat fixed !important;
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        background: url('models/eye_diagram.png') center/cover no-repeat;
        opacity: {opacity};
        z-index: 0;
        pointer-events: none;
    }}
    .stApp > div {{ position: relative; z-index: 1; }}
    </style>
    """, unsafe_allow_html=True)

def page_login():
    """Login page with clean text layout"""
    init_state()  # Initialize state at top of page
    
    st.set_page_config(page_title="Diabetic Retinopathy Detection", layout="wide")
    
    # Initialize session state for login flow
    if 'login_flow_stage' not in st.session_state:
        st.session_state.login_flow_stage = 'landing'
    
    # Set white background
    st.markdown("""
    <style>
    .stApp { 
        background-color: white !important;
        background-image: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ==================== LANDING PAGE ====================
    if st.session_state.login_flow_stage == 'landing':
        # Set white background and button styles
        st.markdown("""
        <style>
        .stApp {
            background-color: white !important;
            background-image: none !important;
        }
        
        [data-testid="stAppViewContainer"] {
            background-color: white !important;
        }
        
        [data-testid="stVerticalBlock"] {
            background-color: white !important;
        }
        
        /* Make right column fill full height */
        [data-testid="column"]:nth-child(2) {
            height: 100vh;
            padding: 0 !important;
            margin: 0 !important;
        }
        
        [data-testid="column"]:nth-child(2) img {
            height: 100%;
            width: 100%;
            object-fit: cover;
            margin: 0;
            padding: 0;
        }
        
        /* Style for both buttons on landing page: dark blue background */
        [data-testid="column"]:nth-child(1) button,
        [data-testid="column"]:nth-child(2) button {
            background-color: #0D47A1 !important;
            color: white !important;
            font-weight: 600 !important;
            font-size: 20px !important;
            padding: 20px 32px !important;
            height: 70px !important;
            border: none !important;
            width: 100% !important;
            max-width: 300px !important;
        }
        [data-testid="column"]:nth-child(1) button:hover,
        [data-testid="column"]:nth-child(2) button:hover {
            background-color: #002171 !important; /* darker blue on hover */
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create layout: text content on left, slightly smaller hero image on right
        left_col, right_col = st.columns([0.8, 1.2], gap="medium")
        
        with left_col:
            # Add space to position title near the top
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            
            # Very large orange title
            st.markdown(
                "<h1 style='font-size: 96px; font-weight: 900; color: #FF6B35; margin: 0; line-height: 1.0;'>"
                "DIABETIC RETINOPATHY</h1>",
                unsafe_allow_html=True
            )
            
            st.write("")
            
            # Orange subtitle (30px size, light red color)
            st.markdown(
                "<p style='font-size: 30px; color: #FF6B6B; margin: 0; font-weight: 400; letter-spacing: 0.5px;'>"
                "Eye Detection System</p>",
                unsafe_allow_html=True
            )
            
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            
            # Buttons side by side with custom styling
            btn_col1, btn_col2 = st.columns(2, gap="small")
            
            with btn_col1:
                if st.button("Register", use_container_width=True, key="btn_landing_register"):
                    st.session_state.login_flow_stage = 'register'
                    st.rerun()
            
            with btn_col2:
                if st.button("Login", use_container_width=True, key="btn_landing_login"):
                    st.session_state.login_flow_stage = 'login'
                    st.rerun()
            
            # additional inline style to ensure buttons are red regardless of surrounding CSS
            st.markdown("""
            <style>
            button[data-key="btn_landing_register"],
            button[data-key="btn_landing_login"] {
                background-color: #D32F2F !important;
                color: white !important;
                border: none !important;
                font-size: 20px !important;
                padding: 20px 32px !important;
                height: 70px !important;
            }
            button[data-key="btn_landing_register"]:hover,
            button[data-key="btn_landing_login"]:hover {
                background-color: #B71C1C !important;
            }
            </style>
            """, unsafe_allow_html=True)
        
        with right_col:
            # Display full-height hero eye image
            try:
                eye_img_path = os.path.join('models', 'eye_real.png')
                if os.path.exists(eye_img_path):
                    eye_img = Image.open(eye_img_path)
                    st.image(eye_img, use_container_width=True)
                else:
                    eye_img_path = os.path.join('models', 'eye_diagram.png')
                    if os.path.exists(eye_img_path):
                        eye_img = Image.open(eye_img_path)
                        st.image(eye_img, use_container_width=True)
            except:
                st.info("Eye Image Area")        
        # Blue-background footer
        st.markdown("""
        <div style='background-color: #E3F2FD; padding: 10px; text-align: center;'>
            Eye Severity Detection System © 2026 | For educational purposes only
        </div>
        """, unsafe_allow_html=True)    
    # ==================== REGISTRATION PAGE (SINGLE STEP) ====================
    elif st.session_state.login_flow_stage == 'register':
        set_app_background(opacity=0.20)
        st.markdown("""
        <style>
        .stApp {
            background-color: rgba(255, 255, 255, 0.98) !important;
        }
        [data-testid="stAppViewContainer"] {
            background-color: rgba(255, 255, 255, 0.98) !important;
        }
        .stTextInput>div>input, .stNumberInput>div>input {
            font-size: 18px !important;
            color: #000 !important;
            padding: 12px !important;
        }
        .stRadio {
            margin-top: 10px !important;
        }
        .stRadio>div>label {
            font-size: 18px !important;
        }
        .stButton>button {
            background-color: #0D47A1 !important;
            color: white !important;
            font-size: 18px !important;
            padding: 12px 24px !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
        }
        .stButton>button:hover {
            background-color: #0A3A7F !important;
            opacity: 1 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Center the form
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Heading
            st.markdown(
                "<h2 style='font-size:48px; color:#FF8C00; text-align:center; margin-bottom:30px;'>"
                "Create New Account</h2>",
                unsafe_allow_html=True
            )
            
            st.write("")  # Spacing
            
            # Form fields
            username = st.text_input(
                "Username",
                placeholder="Choose a username",
                key="reg_username"
            )
            
            name_col1, name_col2 = st.columns(2, gap="small")
            with name_col1:
                first_name = st.text_input(
                    "First Name", 
                    placeholder="First name",
                    key="reg_first_name"
                )
            with name_col2:
                last_name = st.text_input(
                    "Last Name", 
                    placeholder="Last name",
                    key="reg_last_name"
                )
            
            age = st.number_input(
                "Age",
                min_value=0,
                max_value=120,
                key="reg_age"
            )
            
            gender = st.radio(
                "Gender",
                ["Male", "Female", "Other"],
                horizontal=True,
                key="reg_gender"
            )
            
            password = st.text_input(
                "Password",
                type="password",
                placeholder="At least 6 characters",
                key="reg_password"
            )

            
            st.write("")  # Spacing before buttons
            st.write("")
            
            # Buttons - Back on left, Register on right
            col_back, col_register = st.columns(2, gap="medium")
            
            with col_back:
                if st.button("Back", use_container_width=True, key="btn_register_back"):
                    st.session_state.login_flow_stage = 'landing'
                    st.rerun()
            
            with col_register:
                if st.button("Register", use_container_width=True, key="btn_register_submit"):
                    # Validation
                    if not username or not first_name or not last_name or not password:
                        st.error("Please fill all fields")
                    elif not username.isalnum() or len(username) < 3:
                        st.error("Username must be 3+ alphanumeric characters")
                    elif not first_name.replace(" ", "").isalpha():
                        st.error("First name can only contain letters")
                    elif not last_name.replace(" ", "").isalpha():
                        st.error("Last name can only contain letters")
                    elif len(password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        # Proceed with registration
                        from auth import register_user
                        success, message = register_user(
                            username,
                            first_name,
                            last_name,
                            age,
                            gender,
                            password
                        )
                        if success:
                            st.success("✓ Registration successful!")
                            st.info(message)
                            st.write("Redirecting to login page...")
                            st.session_state.login_flow_stage = 'login'
                            import time
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error(message)


    # ==================== LOGIN PAGE ====================
    elif st.session_state.login_flow_stage == 'login':
        set_app_background(opacity=0.20)
        st.markdown("""
        <style>
        .stTextInput>div>input, .stNumberInput>div>input { font-size: 20px !important; color: #000 !important; }
        .stButton>button {
            background-color: #0D47A1 !important;
            color: white !important;
            font-size: 20px !important;
            padding: 12px 24px !important;
            border-radius: 6px !important;
        }
        .stButton>button:hover { opacity: 0.9; }
        </style>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("## Login")
            st.write("")
            
            login_username = st.text_input("Username", key="login_username_input", placeholder="Enter username")
            login_password = st.text_input("Password", type="password", key="login_password_input", placeholder="Enter password")
            
            st.write("")
            
            col_submit, col_back = st.columns(2)
            
            with col_submit:
                if st.button("Login", use_container_width=True, key="btn_login_submit"):
                    from auth import login_user, user_exists
                    
                    if not login_username or not login_password:
                        st.error("Please enter username and password")
                    elif not user_exists(login_username):
                        st.error("No account with this username")
                    elif login_user(login_username, login_password):
                        st.session_state.username = login_username
                        st.session_state.logged_in = True
                        st.session_state.page = "Dashboard"
                        st.success("Successfully Logged In!")
                        # balloons removed per request
                        import time
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
            
            with col_back:
                if st.button("Back", use_container_width=True, key="btn_login_back"):
                    st.session_state.login_flow_stage = 'landing'
                    st.rerun()
        



def page_eye_analysis():
    """Eye Analysis page with retinal image validation"""
    init_state()  # Initialize state at top of page
    
    st.set_page_config(page_title="Eye Analysis", layout="wide")
    set_app_background(opacity=0.25)

    st.title("Eye Analysis")
    st.write("Upload a retinal/eye fundus image for severity analysis")
    
    col1, col2 = st.columns([1, 2])
    
    # allow user to pick between file upload or camera capture
    with col1:
        st.subheader("Upload Image")
        method = st.radio("Input method", ["File upload", "Camera"], index=0)
        uploaded_file = None
        if method == "File upload":
            uploaded_file = st.file_uploader("Choose an eye image", type=['jpg', 'jpeg', 'png', 'bmp'])
        else:
            # camera_input returns an UploadedFile-like object when an image is captured
            cam = st.camera_input("Take a live photo of your eye")
            if cam:
                uploaded_file = cam
    
    with col2:
        st.subheader("Preview")
        if uploaded_file:
            try:
                if isinstance(uploaded_file, np.ndarray):
                    image = Image.fromarray(uploaded_file)
                else:
                    image = Image.open(uploaded_file)
                st.image(image, use_container_width=True, caption="Uploaded Image")
            except Exception:
                st.error("Could not load preview of the selected image.")
    
    st.write("---")
    
    if uploaded_file:
        # Validate image first using permissive API and ensure an eye is present
        is_valid, image_array, message = validate_uploaded_image(uploaded_file)
        from image_validator import detect_eye
        if is_valid and image_array is not None:
            if not detect_eye(image_array):
                is_valid = False
                message = "This image is not supported. Please upload a clear eye image."

        if is_valid:
            st.success(message)

            # Show validation details
            with st.expander("View validation details"):
                details = get_validation_details(uploaded_file)
                st.write(f"**Validation Confidence:** {details['confidence']:.1f}%")
                for detail in details['details']:
                    st.write(detail)

            if st.button("Analyze Eye Image", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    username = get_current_user()
                    users = load_users()
                    profile = users.get(username, {})
                    patient_name = profile.get('name', username)
                    patient_age = profile.get('age', '')

                    # Determine override condition based on stored registration data
                    years_diabetes = profile.get('years_diabetes', None)
                    diabetic_level = str(profile.get('level', '')).strip().lower()

                    try:
                        years_diabetes_val = float(years_diabetes) if years_diabetes is not None else None
                    except Exception:
                        years_diabetes_val = None

                    override_normal = (years_diabetes_val == 0 and diabetic_level == 'normal')
                    is_retinal = is_retinal_image(image_array)

                    # Prepare image
                    if isinstance(uploaded_file, np.ndarray):
                        image = Image.fromarray(uploaded_file)
                        filename = 'camera_capture.jpg'
                    else:
                        image = Image.open(uploaded_file)
                        filename = uploaded_file.name

                    if override_normal:
                        result = {
                            'severity': 'Normal (No Diabetic Retinopathy Detected)',
                            'confidence': 1.0,
                            'features': {},
                            'model_type': 'Override',
                        }
                    else:
                        detector = EyeSeverityDetector()
                        # Prepare patient data for classification
                        patient_data = {
                            'age': patient_age,
                            'years_diabetes': years_diabetes_val,
                            'hba1c': profile.get('hba1c'),
                            'gender': profile.get('gender')
                        }
                        result = detector.classify_severity(image, patient_data)

                    analysis_timestamp = datetime.now().isoformat()
                    safe_base = ''.join(c if c.isalnum() else '_' for c in os.path.splitext(filename)[0])
                    saved_filename = f"{username}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{safe_base}.jpg"
                    image_save_path = os.path.join(UPLOAD_DIR, saved_filename)

                    try:
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        image.save(image_save_path, format='JPEG')
                    except Exception as e:
                        print(f"Could not save analysis image: {e}")
                        image_save_path = None

                    # SAVE TO ANALYSIS HISTORY IMMEDIATELY (CRITICAL FIX)
                    new_entry = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "severity": result["severity"].lower().strip(),
                        "confidence": float(result["confidence"]) * 100,
                        "filename": filename
                    }
                    
                    # SAVE DATA (CRITICAL)
                    st.session_state.analysis_history.append(new_entry)
                    
                    # SAVE LATEST RESULT
                    st.session_state.latest_result = new_entry
                    
                    # DEBUG (TEMPORARY)
                    st.write("DEBUG:", st.session_state.analysis_history)

                    st.session_state.last_analysis = {
                        'username': username,
                        'image': image,
                        'filename': filename,
                        'result': result,
                        'patient_name': patient_name,
                        'patient_age': patient_age,
                        'analysis_timestamp': analysis_timestamp,
                        'image_path': image_save_path
                    }

                    saved_prediction = save_prediction(
                        username,
                        filename,
                        result['severity'],
                        result['confidence'],
                        result.get('features', {}),
                        image_path=image_save_path,
                        predictions=result.get('predictions', {})
                    )
                    if saved_prediction is not None:
                        st.session_state.last_analysis['id'] = saved_prediction.get('id')
                        st.session_state.last_analysis['confidence'] = saved_prediction.get('confidence', result['confidence'])
                        st.session_state.dashboard_refresh = True

                    # Force navigation to Analysis Result page
                    st.session_state.page = "Analysis Result"
                    print(f"✓ Analysis complete. Navigating to Analysis Result. Page state: {st.session_state.page}")
                    st.rerun()
        else:
            # provide clearer message for unsupported photos with custom styling
                st.markdown(
                    "<p style='color: red; font-weight: bold; font-size:20px;'>"
                    "Please upload a valid retinal eye image."
                    "</p>", unsafe_allow_html=True)
                st.info("Valid eye photos should clearly show an eye (iris or fundus)")
                st.write("- Keep the eye centered and in focus")
                st.write("- Avoid portraits or non-eye subjects")
                st.write("- Good lighting helps detection")

def page_analysis_result():
    """Separate page showing the latest analysis result with download option"""
    init_state()  # Initialize state at top of page
    
    st.set_page_config(page_title="Analysis Result", layout="wide")
    set_app_background(opacity=0.25)

    if 'last_analysis' not in st.session_state:
        st.info("No analysis available. Please upload an image first.")
        if st.button("Go to Eye Analysis"):
            st.session_state.page = "Eye Analysis"
            st.rerun()
        return

    data = st.session_state.last_analysis
    image = data['image']
    filename = data['filename']
    result = data['result']
    severity = result['severity']
    confidence = data.get('confidence', result['confidence'])
    if confidence <= 1:
        confidence = confidence * 100
    analysis_timestamp = data.get('analysis_timestamp')

    st.title("Analysis Result")
    
    # Map severity to class number for display
    severity_class_map = {
        'normal': 0,
        'mild': 1,
        'moderate': 2,
        'severe': 3,
        'proliferative': 4
    }
    severity_normalized = normalize_severity_label(severity)
    class_num = severity_class_map.get(severity_normalized, 0)
    severity_display_map = {
        'normal': 'Normal',
        'mild': 'Mild',
        'moderate': 'Moderate',
        'severe': 'Severe',
        'proliferative': 'Proliferative'
    }
    severity_name = severity_display_map.get(severity_normalized, severity.title())
    
    date_display = "Unknown"
    if analysis_timestamp:
        try:
            date_display = datetime.fromisoformat(analysis_timestamp).strftime('%B %d, %Y %I:%M %p')
        except Exception:
            date_display = analysis_timestamp

    st.markdown("### Result Summary")
    st.markdown(f"**File Name:** {filename}")
    st.markdown(f"**Date:** {date_display}")
    st.markdown(f"**Severity:** {severity_name}")
    st.markdown(f"**Confidence:** {confidence:.2f}%")

    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, use_container_width=True)
    with col2:
        color = get_severity_color(severity_normalized)
        explanation = result.get('explanation', 'Analysis completed')
        st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: {color}20; border-left: 4px solid {color};">
                <h3 style="color: {color};">Prediction: Class {class_num} – {severity_name}</h3>
                <p><strong>Confidence Score:</strong> {confidence*100:.2f}%</p>
                <p><strong>Explanation:</strong> {explanation}</p>
            </div>
        """, unsafe_allow_html=True)
        
        detector = EyeSeverityDetector()
        recs = detector.get_recommendations(severity_normalized)

        # PDF generation using BytesIO for proper bytes format
        from fpdf import FPDF
        from io import BytesIO

        # Use stored patient info if available
        patient_name = data.get('patient_name', '') or 'Not provided'
        patient_age = data.get('patient_age', '') or 'Not provided'
        analysis_timestamp = data.get('analysis_timestamp') or datetime.now().isoformat()
        analysis_datetime = datetime.fromisoformat(analysis_timestamp).strftime('%m/%d/%Y %H:%M:%S')

        def gen_pdf():
            # Create PDF with BytesIO buffer
            buffer = BytesIO()
            pdf = FPDF(orientation='P', unit='mm', format='A4')
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()

            # TITLE: centered, bold
            pdf.set_font("Arial", "B", 18)
            pdf.cell(0, 10, "EYE ANALYSIS REPORT", ln=True, align='C')
            pdf.ln(2)

            # Patient details in ONE LINE
            pdf.set_font("Arial", size=10)
            safe_patient_name = str(patient_name)[:40] if patient_name else 'Not provided'
            safe_age = str(patient_age) if patient_age not in (None, '', 0) else 'Not provided'
            patient_line = f"NAME: {safe_patient_name}     AGE: {safe_age}     DATE: {analysis_datetime}"
            pdf.cell(0, 6, patient_line, ln=True)
            pdf.ln(4)

            # Image
            try:
                temp_path = os.path.join('models', 'temp_eye_image.jpg')
                image.save(temp_path)
                img_width = 120
                x_center = (pdf.w - img_width) / 2
                pdf.image(temp_path, x=x_center, w=img_width)
                pdf.ln(6)
            except Exception:
                pdf.ln(6)

            # Severity level
            pdf.set_font("Arial", "B", 12)
            safe_severity = severity_name.encode('ascii', 'replace').decode('ascii')
            pdf.cell(0, 8, f"SEVERITY LEVEL: {safe_severity}", ln=True)
            pdf.ln(3)

            # Recommendations with line spacing
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 8, "RECOMMENDATIONS:", ln=True)
            pdf.ln(2)
            pdf.set_font("Arial", size=10)
            
            # Get up to 4 recommendations
            recommendations = recs if recs else [
                "Schedule an eye examination with an ophthalmologist",
                "Maintain proper blood glucose control",
                "Regular eye screening is recommended",
                "Follow doctor's advice and medication"
            ]
            
            page_width = pdf.w - pdf.l_margin - pdf.r_margin
            for i, rec in enumerate(recommendations[:4], 1):
                safe_rec = rec.encode('ascii', 'replace').decode('ascii')
                pdf.multi_cell(page_width, 6, f"{i}. {safe_rec}")
                pdf.ln(1)

            # Output to bytes
            pdf_bytes = pdf.output(dest='S')
            
            # Ensure it's bytes, not bytearray
            if isinstance(pdf_bytes, bytearray):
                pdf_bytes = bytes(pdf_bytes)
            
            return pdf_bytes

        pdf_bytes = gen_pdf()
        
        st.markdown("---")
        st.download_button(
            label="DOWNLOAD RECEIPT",
            data=pdf_bytes,
            file_name="analysis_receipt.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="download_receipt_button"
        )

        # Back to Dashboard button
        st.markdown("---")
        if st.button("Back to Dashboard", use_container_width=True):
            st.session_state.page = "Dashboard"
            st.rerun()

def page_dashboard():
    """Dashboard page showing summary statistics and recent history."""
    init_state()  # Initialize state at top of page
    
    st.set_page_config(page_title="Dashboard", layout="wide")
    set_app_background(opacity=0.25)

    username = get_current_user()
    
    # Reset dashboard refresh flag
    if st.session_state.get('dashboard_refresh'):
        st.session_state.dashboard_refresh = False

    # Read from session state analysis_history
    data = st.session_state.analysis_history
    
    # Calculate statistics from session data
    total_analyses = len(data)
    user_analyses = len(data)
    
    avg_confidence = 0
    if total_analyses > 0:
        avg_confidence = sum(d["confidence"] for d in data) / total_analyses

    # Get latest analysis for image display (full object)
    latest_analysis = st.session_state.get('last_analysis')
    
    # Get latest result for display (from analysis_history)
    latest_result = st.session_state.get('latest_result')

    # Professional header
    st.title("📊 Diabetic Retinopathy Detection Dashboard")
    st.markdown(f"**Welcome back, {username}!** Here's your analysis overview.")

    # Top metric row - ALWAYS SHOW
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1], gap="large")
    with col1:
        st.markdown("#### Total Analyses")
        st.markdown(f"### {total_analyses}")
    with col2:
        st.markdown("#### User Analyses")
        st.markdown(f"### {user_analyses}")
    with col3:
        st.markdown("#### Avg Confidence")
        st.markdown(f"### {avg_confidence:.2f}%")
    with col4:
        st.markdown("#### System Status")
        st.markdown("### Active ✓")

    st.markdown("---")

    # Chart and latest analysis section - ALWAYS SHOW
    chart_col, right_col = st.columns([2, 1], gap="large")

    with chart_col:
        st.markdown("### 📊 Diabetic Retinopathy Severity Distribution")
        severity_levels = ["normal", "mild", "moderate", "severe", "proliferative"]
        counts = {level: 0 for level in severity_levels}

        for d in data:
            sev = d["severity"].lower().strip()
            if sev in counts:
                counts[sev] += 1

        # Always show graph, even if empty
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#00AA00', '#FFFF00', '#FFA500', '#FF6600', '#8B0000']
        bars = ax.bar(counts.keys(), counts.values(), color=colors, alpha=0.9)
        ax.set_ylabel("Count", fontsize=12, fontweight='bold')
        ax.set_xlabel("Severity Level", fontsize=12, fontweight='bold')
        ax.set_title("Severity Distribution by Your Analyses", fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.25)
        ax.set_ylim(bottom=0)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1, str(int(height)), 
                       ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.xticks(rotation=0)
        st.pyplot(fig)

    with right_col:
        st.markdown("### 📸 Latest Analysis")
        if data:  # Use data from analysis_history
            latest_entry = data[-1]  # Get most recent entry
            # Get image from last_analysis if available
            last_analysis_obj = st.session_state.get('last_analysis')
            if last_analysis_obj and last_analysis_obj.get('image') is not None:
                st.image(last_analysis_obj['image'], use_container_width=True)
            st.markdown(f"**File:** {latest_entry.get('filename', 'Unknown')}  ")
            st.markdown(f"**Severity:** {latest_entry['severity'].title()}  ")
            st.markdown(f"**Confidence:** {latest_entry['confidence']:.2f}%")
            st.markdown(f"**Time:** {latest_entry['timestamp']}")
        else:
            st.info("No recent analysis available. Start by uploading an eye image.")

        st.markdown("---")
        st.markdown("### 📋 User Analysis History")
        
        if data:
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, height=400)
        else:
            st.warning("No analysis history yet")

    st.markdown("---")

    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📹 New Analysis", use_container_width=True):
            st.session_state.page = "Eye Analysis"
            st.rerun()
    with col2:
        if st.button("📋 View Detailed History", use_container_width=True):
            st.session_state.page = "History"
            st.rerun()
    with col3:
        if st.button("🚪 Logout", use_container_width=True):
            logout_user()
            st.rerun()


def page_history():
    """History page"""
    init_state()  # Initialize state at top of page
    
    st.set_page_config(page_title="Analysis History", layout="wide")
    set_app_background(opacity=0.25)

    st.title("Analysis History")
    
    username = get_current_user()
    predictions = get_user_predictions(username)
    
    if predictions:
        df = pd.DataFrame(predictions)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            severity_filter = st.multiselect(
                "Filter by Severity",
                options=list(SEVERITY_LEVELS.keys()),
                default=list(SEVERITY_LEVELS.keys())
            )

        df_filtered = df[df['severity'].isin(severity_filter)]
        st.write(f"Showing {len(df_filtered)} of {len(df)} analyses")
        st.write("---")

        df_display = df_filtered[['timestamp', 'severity', 'confidence']].copy()
        df_display['timestamp'] = df_display['timestamp'].str[:19]
        df_display['severity'] = df_display['severity'].str.title()
        df_display['confidence'] = df_display['confidence'].astype(str) + '%'
        df_display = df_display.rename(columns={
            'timestamp': 'Timestamp',
            'severity': 'Severity',
            'confidence': 'Confidence'
        })
        df_display = df_display.sort_values('Timestamp', ascending=False)
        st.dataframe(df_display, use_container_width=True, height=520)
    else:
        st.info("No analysis history yet")
    
    st.write("---")
    
    if st.button("Back to Dashboard"):
        st.session_state.page = "Dashboard"
        st.rerun()
