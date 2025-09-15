import streamlit as st
from utils import log_action

def social_page():
    st.title("📱 Social Media Verticle")
    st.write("Twitter, Instagram, YouTube, Reddit, Quora")

    if st.button("Twitter"):
        log_action(st.session_state["username"], "Open", "Social Media → Twitter")
        st.info("🔹 Twitter module coming soon.")

    if st.button("Instagram"):
        log_action(st.session_state["username"], "Open", "Social Media → Instagram")
        st.info("🔹 Instagram module coming soon.")

if "authenticated" in st.session_state and st.session_state["authenticated"]:
    social_page()
else:
    st.error("Please login first.")
