import streamlit as st
import os
import subprocess
import sys


def signup_page():
    """
    Renders the Sign-Up page for users to create their profiles and upload images.
    """
    st.header("Sign Up")

    # Create a dataset folder if it doesn't exist
    if not os.path.exists("../dataset"):
        os.makedirs("../dataset")
        st.success("Dataset folder created.")

    # Initialize session state for embedding status
    if "embedding_success" not in st.session_state:
        st.session_state["embedding_success"] = False

    username = st.text_input("Enter your username")
    if username:
        user_folder = os.path.join("../dataset", username)
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)
            st.success(f"Folder created for user: {username}")

        # Upload images
        uploaded_files = st.file_uploader("Upload your images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Save the uploaded file to the user's folder
                with open(os.path.join(user_folder, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Saved {uploaded_file.name} to {user_folder}")

            # Run the embedding script
            if st.button("User Register"):
                try:
                    result = subprocess.run(
                        [sys.executable, "generate_face_embeddings.py"],
                        capture_output=True, text=True, check=True
                    )
                    st.success("User Registration successfully!")
                    st.write(result.stdout)
                    st.session_state["embedding_success"] = True  # Update session state

                except subprocess.CalledProcessError as e:
                    st.error("Error while running the embedding script.")
                    st.write(e.stderr)

    # Display "Go to Log In" button only if embeddings were generated successfully
    if st.session_state["embedding_success"]:
        st.button("Go to Log In", on_click=lambda: st.session_state.update(page="login"))
