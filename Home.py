# import streamlit as st
# from utils import log_action

# # Hardcoded admin credentials
# ADMINS = {
#     "test": "rsm@123",
#     "admin": "rsm@123",
#     "user": "rsm@123"
# }

# def login_page():
#     st.title("🔐 Login")

#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")

#     if st.button("Login"):
#         if username in ADMINS and ADMINS[username] == password:
#             st.session_state["authenticated"] = True
#             st.session_state["username"] = username
#             log_action(username, "Login", "SUCCESS")
#             st.success("Login successful!")
#             st.rerun()
#         else:
#             log_action(username, "Login", "FAILED")
#             st.error("Invalid username or password.")

# def main_menu():
#     st.sidebar.title("Navigation")
#     st.sidebar.write(f"👋 Welcome, {st.session_state['username']}")

#     if st.sidebar.button("Logout"):
#         log_action(st.session_state["username"], "Logout", "SUCCESS")
#         st.session_state.clear()
#         st.rerun()

#     st.title("Main Dashboard")
#     st.write("Choose a vertical from the sidebar (pages will appear here).")

# # ==== App Routing ====
# if "authenticated" not in st.session_state:
#     login_page()
# elif not st.session_state["authenticated"]:
#     login_page()
# else:
#     main_menu()




# Home.py
import streamlit as st
import datetime

# --- Streamlit Page Config (set ONCE here only) ---
st.set_page_config(
    page_title="Admin Dashboard",
    page_icon="🔐",
    layout="wide"
)


# --- Hardcoded admins ---
ADMINS = {
    "test": "rsm@123",
    "admin": "rsm@123",
    "user": "rsm@123"
}

# --- Log function ---
def log_action(user, action):
    with open("logs.txt", "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {user}: {action}\n")

# --- Initialize session state ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None

st.title("Admin Login")

# --- If not logged in, show login form ---
if not st.session_state.authenticated:
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if username in ADMINS and ADMINS[username] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            log_action(username, "logged in")
            st.success("Login successful. Please use the sidebar to navigate.")
            st.rerun()
        else:
            st.error("Invalid username or password")

# --- If logged in, show dashboard ---
else:
    st.success(f"Welcome, {st.session_state.username} ✅")
    st.write("Use the sidebar to navigate to different verticals.")

    if st.button("Logout"):
        log_action(st.session_state.username, "logged out")
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()
