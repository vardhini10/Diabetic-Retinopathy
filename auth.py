# Authentication Module

import streamlit as st
from config import USERS
import hashlib
import json
import os

USERS_FILE = os.path.join(os.path.dirname(__file__), 'users.json')

def load_users():
    """Load users from JSON file"""
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r') as f:
                data = json.load(f)
                return data.get('users', {})
    except Exception as e:
        print(f"Error loading users: {e}")
    return USERS

def save_users(users_dict):
    """Save users to JSON file"""
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump({'users': users_dict}, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving users: {e}")
        return False

def hash_password(password):
    """Hash password for security"""
    return hashlib.sha256(password.encode()).hexdigest()

def validate_password(password):
    """Validate password strength requirements"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least 1 capital letter"
    
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least 1 digit"
    
    # Check for special characters (anything that's not letter or digit)
    if not any(not c.isalnum() for c in password):
        return False, "Password must contain at least 1 special character"
    
    return True, "Password is valid"

def register_user(username, first_name, last_name, age, gender, password):
    """Register a new user with username + profile fields"""
    users = load_users()
    
    # Validation
    if not username or not first_name or not last_name or not password:
        return False, "All fields are required"
    
    if len(username) < 3:
        return False, "Username must be at least 3 characters"
    
    # Username can contain letters, numbers, and special characters
    if username in users:
        return False, "Username already exists"
    
    # Validate password strength
    password_valid, password_message = validate_password(password)
    if not password_valid:
        return False, password_message
    
    if not first_name.replace(" ", "").isalpha():
        return False, "First name can only contain letters"
    
    if not last_name.replace(" ", "").isalpha():
        return False, "Last name can only contain letters"
    
    # Add new user with profile data
    users[username] = {
        "password": password,
        "first_name": first_name,
        "last_name": last_name,
        "age": age,
        "gender": gender,
        "name": f"{first_name} {last_name}"  # Full name for display
    }
    
    # Save to file
    if save_users(users):
        return True, "Registration successful! You can now login with your username and password."
    else:
        return False, "Error saving user. Please try again."

def login_user(username, password):
    """Verify user credentials by username"""
    users = load_users()
    if username in users:
        user_data = users[username]
        if isinstance(user_data, dict):
            if user_data.get("password") == password:
                return True
        elif user_data == password:  # Old format compatibility
            return True
    return False

def user_exists(username):
    """Check if user account exists"""
    users = load_users()
    return username in users

def initialize_session():
    """Initialize session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'page' not in st.session_state:
        st.session_state.page = "Dashboard"
    if 'login_flow_stage' not in st.session_state:
        st.session_state.login_flow_stage = 'landing'
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []

def logout_user():
    """Logout the current user and go to landing page"""
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.login_flow_stage = 'landing'

def is_logged_in():
    """Check if user is logged in"""
    return st.session_state.get('logged_in', False)

def get_current_user():
    """Get the current logged-in username"""
    return st.session_state.get('username', None)
