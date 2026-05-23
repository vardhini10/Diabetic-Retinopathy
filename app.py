# Main Streamlit Application - Eye Severity Detection System

import socket
import streamlit as st
from streamlit_option_menu import option_menu
from auth import initialize_session, is_logged_in, get_current_user, logout_user
from pages import page_login, page_eye_analysis, page_dashboard, page_history, page_analysis_result
from config import APP_NAME, APP_VERSION
from state import init_state


def get_local_address(port: int = 8501) -> str: 
    """Return a string users can use to access the running Streamlit app."""
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        return f"http://{local_ip}:{port}"
    except Exception:
        # fallback
        return f"http://localhost:{port}"

print("Local address for accessing the app:", get_local_address())


initialize_session()
init_state()  

# Set page configuration
st.set_page_config(
    page_title="Eye Severity Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Check if user is logged in
if not is_logged_in():
    # Show login page
    page_login()
else:
    # Logged in - show main application
    username = get_current_user()
    
    # Sidebar with navigation
    with st.sidebar:
        st.markdown(f"## 👁️ {APP_NAME}")
        st.markdown(f"Version: {APP_VERSION}")
        # show local URL so users know where to connect
        try:
            local_addr = get_local_address()
            st.markdown(f"**URL:** {local_addr}")
        except Exception:
            pass
        st.write("---")
        
        st.write(f"**Logged in as:** {username}")
        st.write("---")
        
        
        menu_options = ["Dashboard", "Eye Analysis", "History", "Logout"]
        menu_icons = ["speedometer2", "eye", "clock-history", "box-arrow-right"]
    
        if 'last_analysis' in st.session_state:
            menu_options.insert(2, "Analysis Result")
            menu_icons.insert(2, "journal-text")
        try:
            default_idx = menu_options.index(st.session_state.page) if st.session_state.page in menu_options else 0
        except Exception:
            default_idx = 0

        selected = option_menu(
            menu_title="Navigation",
            options=menu_options,
            icons=menu_icons,
            menu_icon="cast",
            default_index=default_idx,
        )
        
        # Handle navigation
        if selected == "Dashboard":
            st.session_state.page = "Dashboard"
        elif selected == "Eye Analysis":
            st.session_state.page = "Eye Analysis"
        elif selected == "Analysis Result":
            st.session_state.page = "Analysis Result"
        elif selected == "History":
            st.session_state.page = "History"
        elif selected == "Logout":
            logout_user()
            st.rerun()
        
        st.write("---")
        
        # Sidebar info
        with st.expander("ℹ️ About this System"):
            st.info("""
            This system uses advanced image analysis to detect 
            diabetic retinopathy severity in eye images.
            
            **Severity Levels:**
            - 🟢 Normal
            - 🟡 Mild
            - 🟠 Moderate
            - 🔴 Severe
            - ⚫ Proliferative
            """)
    

    if st.session_state.page == "Dashboard":
        page_dashboard()
    elif st.session_state.page == "Eye Analysis":
        page_eye_analysis()
    elif st.session_state.page == "Analysis Result":
        
        page_analysis_result()
    elif st.session_state.page == "History":
        page_history()


