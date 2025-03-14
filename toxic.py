import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import speech_recognition as sr
import os

try:
    # Load the model
    model_path = os.path.join(os.getcwd(), 'toxic_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load the vectorizer
    vectorizer_path = os.path.join(os.getcwd(), 'vectorizer.pkl')
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    print("Model and vectorizer loaded successfully.")

except FileNotFoundError as e:
    print(f"Error: {e}. Ensure that the files are in the correct directory.")
except PermissionError as e:
    print(f"Error: {e}. Check file permissions.")
except Exception as e:
    print(f"An error occurred: {e}")

# Title and layout
st.title('Toxic Comment Detection')
st.write("### Enter a comment below:")

# Initialize session state for comment
if 'comment' not in st.session_state:
    st.session_state.comment = ""

# Input box for comment (binds to session state)
comment = st.text_area("Comment", st.session_state.comment)

# Initialize Speech Recognition
recognizer = sr.Recognizer()

# Function to predict toxicity
def predict_toxicity(comment):
    comment_cleaned = [comment.lower()]
    comment_vectorized = vectorizer.transform(comment_cleaned)
    prediction = model.predict(comment_vectorized)[0]
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    result = {}
    for i, label in enumerate(labels):
        result[label] = prediction[i]
    
    return result

# Speech-to-Text Button (automatically updates text box)
if st.button("🎙️ Speak"):
    with sr.Microphone() as source:
        st.write("Listening...")
        try:
            audio = recognizer.listen(source, timeout=5)
            recognized_text = recognizer.recognize_google(audio)
            st.session_state.comment = recognized_text  # Update session state
            st.success(f"Recognized: {recognized_text}")
            
            # Trigger toxicity check automatically
            result = predict_toxicity(st.session_state.comment)
            st.write("### Prediction Results:")
            for label, value in result.items():
                st.write(f"**{label}**: {'Yes' if value == 1 else 'No'}")
        
        except sr.UnknownValueError:
            st.error("Sorry, could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
        except Exception as e:
            st.error(f"Error: {e}")

# Manual Prediction Button
if st.button("Check Toxicity"):
    if comment.strip() == "":
        st.warning("Please enter a comment!")
    else:
        result = predict_toxicity(comment)
        st.write("### Prediction Results:")
        for label, value in result.items():
            st.write(f"**{label}**: {'Yes' if value == 1 else 'No'}")

