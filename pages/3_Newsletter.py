import streamlit as st
from utils import log_action

def newsletter_page():
    st.title("📰 Newsletter Verticle")
    st.write("Substack and Medium article modules")

    if st.button("Substack"):
        log_action(st.session_state["username"], "Open", "Newsletter → Substack")
        st.info("🔹 Substack module coming soon.")

    if st.button("Medium"):
        log_action(st.session_state["username"], "Open", "Newsletter → Medium")
        st.info("🔹 Medium module coming soon.")

if "authenticated" in st.session_state and st.session_state["authenticated"]:
    newsletter_page()
else:
    st.error("Please login first.")
