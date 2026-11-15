# Streamlit interface: enter an SMS-style message and see whether the saved model classifies it as spam or not spam.
import streamlit as st

from src.spam_detector.prediction import load_model, predict_message

st.set_page_config(page_title="SMS Spam Detector")


@st.cache_resource
def get_model():
    """Load the saved pipeline once and reuse it across reruns."""
    return load_model()


st.title("SMS Spam Detector")
st.write("Enter a text message to see whether the model classifies it as spam.")

# Stop with a clear message if the model file is missing.
try:
    model = get_model()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

message = st.text_area("Message", height=150, placeholder="Type or paste a message here...")

if st.button("Check message"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        result = predict_message(message, model)
        if result["label"] == "spam":
            st.error("Prediction: Spam")
        else:
            st.success("Prediction: Not Spam")

        st.metric("Decision score", f"{result['decision_score']:.2f}")
        st.caption(
            "The score is the distance from the model's decision boundary: above 0 means spam, "
            "below 0 means not spam, and values near 0 are borderline. It is not a probability."
        )

st.divider()
st.caption(
    "This is a machine-learning classifier and can make mistakes; do not rely on it alone for important "
    "messages. It was trained on the UCI SMS Spam Collection and has not been validated on WhatsApp or "
    "other kinds of messages. On a held-out test set of 1,034 messages, it wrongly flagged 1 of 903 "
    "legitimate messages and caught 122 of 131 spam messages."
)