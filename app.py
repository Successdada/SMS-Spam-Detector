# Streamlit interface with a custom design: enter an SMS-style message and see the model's spam / not spam classification.
import streamlit as st

from src.spam_detector.prediction import load_model, predict_message

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")

SPAM_EXAMPLE = "WINNER! You have won a free prize. Call 09061701461 now to claim."
HAM_EXAMPLE = "Are you coming to dinner tonight?"

CUSTOM_CSS = """
<style>
footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 760px;}
.hero {background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 100%); padding: 2rem 1.5rem;
       border-radius: 16px; color: #ffffff; margin-bottom: 1.5rem;}
.hero h1 {margin: 0; padding: 0; font-size: 2rem; color: #ffffff;}
.hero p {margin: 0.4rem 0 0; opacity: 0.9; color: #ffffff;}
.result {padding: 1.25rem 1.5rem; border-radius: 14px; margin: 1rem 0 0.5rem;}
.result.spam {background: #fee2e2; border: 1px solid #fca5a5; color: #991b1b;}
.result.ham {background: #dcfce7; border: 1px solid #86efac; color: #166534;}
.result .label {font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.8;}
.result .verdict {font-size: 1.9rem; font-weight: 700; margin: 0.1rem 0;}
.result .score {font-size: 0.95rem;}
.meter-track {position: relative; height: 10px; border-radius: 999px; margin-top: 1rem;
              background: linear-gradient(90deg, #22c55e 0%, #e5e7eb 50%, #ef4444 100%);}
.meter-mid {position: absolute; left: 50%; top: -3px; width: 2px; height: 16px; background: #374151;}
.meter-marker {position: absolute; top: -5px; width: 18px; height: 18px; margin-left: -9px;
               border-radius: 50%; background: #111827; border: 3px solid #ffffff;
               box-shadow: 0 1px 4px rgba(0, 0, 0, 0.35);}
.meter-labels {display: flex; justify-content: space-between; font-size: 0.75rem; margin-top: 0.5rem; opacity: 0.8;}
.note {font-size: 0.8rem; color: inherit; background: rgba(127, 127, 127, 0.12); border-radius: 12px;
       padding: 0.9rem 1rem; margin-top: 1.5rem;}
div.stButton > button {border-radius: 10px;}
</style>
"""


@st.cache_resource
def get_model():
    """Load the saved pipeline once and reuse it across reruns."""
    return load_model()


def use_example(text):
    """Button callback: put an example message into the text box."""
    st.session_state["message_input"] = text


def result_card_html(label, score):
    """Build the coloured result card. Only the label and the numeric score are inserted, never user text."""
    is_spam = label == "spam"
    css_class = "spam" if is_spam else "ham"
    verdict = "Spam" if is_spam else "Not Spam"
    # Marker position on a bar that shows -2 to +2 (display only; scores beyond that sit at the ends).
    position = (max(-2.0, min(2.0, score)) + 2.0) / 4.0 * 100
    return (
        f'<div class="result {css_class}">'
        f'<div class="label">Prediction</div>'
        f'<div class="verdict">{verdict}</div>'
        f'<div class="score">Decision score: {score:+.2f}</div>'
        f'<div class="meter-track"><div class="meter-mid"></div>'
        f'<div class="meter-marker" style="left:{position:.1f}%"></div></div>'
        f'<div class="meter-labels"><span>Not Spam</span><span>Boundary (0)</span><span>Spam</span></div>'
        f"</div>"
    )


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="hero"><h1>SMS Spam Detector</h1>'
    "<p>Paste a text message and see whether the model classifies it as spam.</p></div>",
    unsafe_allow_html=True,
)

# Stop with a clear message if the model file is missing.
try:
    model = get_model()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

with st.container(border=True):
    st.text_area("Message", key="message_input", height=140, placeholder="Type or paste a message here...")
    col_spam, col_ham = st.columns(2)
    col_spam.button("Try a spam-style example", on_click=use_example, args=(SPAM_EXAMPLE,))
    col_ham.button("Try a normal message", on_click=use_example, args=(HAM_EXAMPLE,))
    check = st.button("Check message", type="primary")

if check:
    message = st.session_state.get("message_input", "")
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        result = predict_message(message, model)
        st.markdown(result_card_html(result["label"], result["decision_score"]), unsafe_allow_html=True)
        st.caption(
            "The score is the distance from the model's decision boundary: above 0 means spam, below 0 means "
            "not spam, and values near 0 are borderline. It is not a probability. The bar is clipped to -2 to +2 "
            "for display."
        )

st.markdown(
    '<div class="note">This is a machine-learning classifier and can make mistakes; do not rely on it alone '
    "for important messages. It was trained on the UCI SMS Spam Collection and has not been validated on "
    "WhatsApp or other kinds of messages. On a held-out test set of 1,034 messages, it wrongly flagged 1 of "
    "903 legitimate messages and caught 122 of 131 spam messages.</div>",
    unsafe_allow_html=True,
)