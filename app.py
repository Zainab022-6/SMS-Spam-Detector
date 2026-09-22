import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="centered"
)

st.title("📱 SMS Spam Detector")
st.write("Enter an SMS message below to check whether it is Spam or Ham.")


# -----------------------------
# Load dataset
# -----------------------------
@st.cache_resource
def train_model():

    df = pd.read_excel("spam.xlsx")

    # Keep only required columns
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]

    # Remove missing values
    df = df.dropna()

    # Convert labels
    df["label"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    X = df["message"].astype(str)
    y = df["label"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Train model
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    return model, vectorizer


model, vectorizer = train_model()


# -----------------------------
# Message input
# -----------------------------
message = st.text_area(
    "✉️ Enter your SMS:",
    placeholder="Example: Congratulations! You have won a free prize!"
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Check Message"):

    if not message.strip():
        st.warning("Please enter a message.")

    else:
        message_tfidf = vectorizer.transform([message])
        prediction = model.predict(message_tfidf)[0]

        if prediction == 1:
            st.error("🚨 SPAM MESSAGE")
        else:
            st.success("✅ HAM — Normal Message")
