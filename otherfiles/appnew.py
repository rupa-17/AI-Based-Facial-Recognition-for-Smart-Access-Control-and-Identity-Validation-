import streamlit as st
import numpy as np
from PIL import Image
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
appnew.py lastapp.py newapp.py realapp.py
st.title("Temporary Face Registration & Verification")

# Temporary in-memory storage
if "face_db" not in st.session_state:
    st.session_state.face_db = {}

option = st.radio("Choose Option", ["Register", "Verify"])

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)

    if option == "Register":
        name = st.text_input("Enter Name")

        if st.button("Register User"):
            embedding = DeepFace.represent(
                image,
                model_name="Facenet",
                enforce_detection=True
            )[0]["embedding"]

            st.session_state.face_db[name] = embedding
            st.success(f"{name} registered successfully!")

    elif option == "Verify":
        if st.button("Verify User"):

            if len(st.session_state.face_db) == 0:
                st.warning("No users registered yet!")
            else:
                test_embedding = DeepFace.represent(
                    image,
                    model_name="Facenet",
                    enforce_detection=True
                )[0]["embedding"]

                best_match = None
                best_score = -1

                for name, stored_embedding in st.session_state.face_db.items():
                    similarity = cosine_similarity(
                        [test_embedding],
                        [stored_embedding]
                    )[0][0]

                    if similarity > best_score:
                        best_score = similarity
                        best_match = name

                if best_score > 0.6:  # Threshold
                    st.success(f"Recognized as: {best_match}")
                    st.write(f"Similarity Score: {best_score:.2f}")
                else:
                    st.error("No matching user found")