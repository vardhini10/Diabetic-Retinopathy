import streamlit as st

def init_state():
    """Initialize session state variables - call this at the top of every page"""
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []
    if "latest_result" not in st.session_state:
        st.session_state.latest_result = None