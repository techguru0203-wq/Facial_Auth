import streamlit as st

# -------------------------------------
# Set page configuration
# -------------------------------------
st.set_page_config(
    page_title="Face Recognition System",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# -------------------------------------
# Styling for Buttons
# -------------------------------------
button_style = """
    <style>
        .button-container {
            display: flex;
            justify-content: center;
            flex-direction: column;
            align-items: center;
        }
        .button-container div {
            margin: 10px;
        }
        .stButton button {
            width: 200px;
            height: 50px;
            font-size: 16px;
        }
    </style>
"""
st.markdown(button_style, unsafe_allow_html=True)

# -------------------------------------
# Helper Function for Page Navigation
# -------------------------------------
def set_page(page_name):
    """Set the current page in session state."""
    st.session_state["page"] = page_name
# -------------------------------------
# Welcome Page
# -------------------------------------
def welcome_page():
    """
    Displays the welcome page with options to navigate to the Sign Up or Log In pages.
    """
    st.markdown(
        "<h1 style='text-align: center;'>Welcome to the Face Recognition System</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h3 style='text-align: center;'>Please choose an option:</h3>",
        unsafe_allow_html=True
    )

    # Centered Buttons
    with st.container():
        col1, col2, col3 = st.columns([3, 5, 1])  # Center the buttons using columns
        with col2:
            if st.button("Sign Up", on_click=set_page, args=("signup",)):
                pass
            st.write("")  # Empty space between buttons
            if st.button("Log In", on_click=set_page, args=("login",)):
                pass


# -------------------------------------
# Initialize session state for page navigation
# -------------------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "welcome"

# -------------------------------------
# Page Routing
# -------------------------------------
if st.session_state["page"] == "welcome":
    welcome_page()
elif st.session_state["page"] == "signup":
    from signup import signup_page
    signup_page()
elif st.session_state["page"] == "login":
    from login import login_page
    login_page()