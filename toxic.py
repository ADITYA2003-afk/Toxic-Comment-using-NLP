import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import speech_recognition as sr

# Load the model and vectorizer
model = pickle.load(open('toxic_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

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

