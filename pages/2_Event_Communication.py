import streamlit as st
from utils import log_action
import os
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import google.generativeai as genai

# ================== Event Page ==================
def event_page():
    st.title("Invitation for Satsang - Email Sender")

    # ===== Generate Invitation Function =====
    def generate_invitation(date_for_satsang, address_of_satsang, time_of_satsang,
                            program_title, content_type, selected_template_text=None):
        if selected_template_text:
            prompt = (
                f"You will be given a satsang program with the following details:\n"
                f"Date: {date_for_satsang}\nTime: {time_of_satsang}\nTitle: {program_title}\nAddress: {address_of_satsang}\n\n"
                f"You must generate a formal, respectful, and detailed email of type '{content_type}' using the following template:\n\n"
                f"Template: {selected_template_text}\n\n"
                "The output must strictly follow the tone: formal, polite, disciplined, and grammatically perfect. "
                "The email should be short, crisp and to the point. Don't stretch the email unnecessarily. Avoid casual words, slang, or irrelevant details. "
                "Ensure the structure: greeting → purpose → details → spiritual value → closing. "
                "Generate a proper subject line starting with 'Subject:' in the beginning."
                "You are the invitation email writer for Ramashram Satsang, Mathura; generate a warm, respectful, and personalized invitation email for an upcoming Satsang program based on the provided details (date, time, title, address, and optional template); structure the email as greeting → purpose → program details → spiritual value (peace, light, love, meditation) → closing; write in a formal yet devotional and humanized tone that feels welcoming and inspiring; ensure subject line starts with 'Subject:'; keep it crisp and clear, avoiding unnecessary length; close the email with 'With gratitude,' followed by 'Sanjiv Kumar' and 'Ramashram Satsang.'"
"
            )
        else:
            prompt = (
                f"Generate a formal, respectful, and detailed {content_type} email for a satsang program.\n"
                f"Date: {date_for_satsang}, Time: {time_of_satsang}, Title: {program_title}, Address: {address_of_satsang}.\n"
                "The email should be polite, elaborate, and professional. "
                "Generate a proper subject line starting with 'Subject:' in the beginning."
                "You are the invitation email writer for Ramashram Satsang, Mathura; generate a warm, respectful, and personalized invitation email for an upcoming Satsang program based on the provided details (date, time, title, address, and optional template); structure the email as greeting → purpose → program details → spiritual value (peace, light, love, meditation) → closing; write in a formal yet devotional and humanized tone that feels welcoming and inspiring; ensure subject line starts with 'Subject:'; keep it crisp and clear, avoiding unnecessary length; close the email with 'With gratitude,' followed by 'Sanjiv Kumar' and 'Ramashram Satsang.'"

            )

        # --- Gemini API call ---
        api_key = os.getenv("GENAI_API_KEY")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text

    # ===== Email Sending Function =====
    def send_email(sender_email, sender_password, receiver_email, subject, body, attachments=None):
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = subject

        # Body
        body = body.replace("\xa0", " ").encode("utf-8", "ignore").decode("utf-8")
        msg.attach(MIMEText(body, "plain", "utf-8"))

        # Attachments
        if attachments:
            for file in attachments:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(file.getvalue())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={file.name}")
                msg.attach(part)

        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, receiver_email, msg.as_string())
            return True
        except Exception as e:
            return str(e)

    # ================== PART 1: Generate Email ==================
    st.header("Step 1: Generate Email Content")

    with st.form("content_form"):
        date_for_satsang = st.date_input("Enter date for satsang")
        address_of_satsang = st.text_input("Enter address of satsang")
        time_of_satsang = st.time_input("Enter time of satsang")
        program_title = st.text_input("Enter title of the program")
        content_type = st.selectbox(
            "Select type of email",
            ["Invitation", "Reminder", "Feedback", "Thanks for attending"]
        )

        template_file = st.file_uploader("Upload Template CSV (columns: type, template)", type=["csv"])
        selected_template_text = None

        if template_file is not None:
            df_template = pd.read_csv(template_file)
            if "type" not in df_template.columns or "template" not in df_template.columns:
                st.error("Template CSV must have 'type' and 'template' columns")
            else:
                if content_type in df_template["type"].values:
                    selected_template_text = df_template.loc[
                        df_template["type"] == content_type, "template"
                    ].values[0]
                else:
                    st.warning(f"No template found for type '{content_type}' in CSV.")

        submitted_content = st.form_submit_button("Generate Content")

    if submitted_content:
        email_body = generate_invitation(
            date_for_satsang, address_of_satsang, time_of_satsang,
            program_title, content_type, selected_template_text
        )

        # ---- Extract Subject if AI puts it in body ----
        email_subject = f"Satsang {content_type} - {program_title}"  # default
        lines = email_body.splitlines()
        cleaned_lines = []
        for line in lines:
            if line.strip().lower().startswith("subject:"):
                email_subject = line.replace("Subject:", "").strip()
            else:
                cleaned_lines.append(line)
        email_body = "\n".join(cleaned_lines).strip()

        # Save once in session_state
        st.session_state["email_subject"] = email_subject
        st.session_state["generated_email"] = email_body

    # Editable fields always visible if generated
    if "generated_email" in st.session_state:
        st.subheader("Edit Generated Email Before Sending")
        st.session_state["email_subject"] = st.text_input(
            "Email Subject", value=st.session_state["email_subject"], key="subject_edit"
        )
        st.session_state["generated_email"] = st.text_area(
            "Email Body", value=st.session_state["generated_email"], height=400, key="body_edit"
        )

    # ================== PART 2: Send Emails ==================
    st.header("Step 2: Send Emails")

    with st.form("send_form"):
        uploaded_file = st.file_uploader("Upload CSV with recipient emails (columns: email, name [optional])", type=["csv"])
        attachments = st.file_uploader("Upload Attachments", type=None, accept_multiple_files=True)
        sender_email = st.text_input("Your Email Address")
        sender_password = st.text_input("Your Email Password", type="password")
        personalize = st.checkbox("Personalize with recipient names if available")

        submitted_send = st.form_submit_button("Send Emails")

    if submitted_send:
        if "generated_email" not in st.session_state:
            st.error("Please generate the email content in Step 1 first.")
        elif uploaded_file is None:
            st.error("Please upload a CSV file with emails.")
        else:
            df = pd.read_csv(uploaded_file)
            if "email" not in df.columns:
                st.error("CSV must contain a column named 'email'")
            else:
                email_subject = st.session_state["email_subject"]
                email_body = st.session_state["generated_email"]

                successes, failures = [], []
                for _, row in df.iterrows():
                    recipient = row["email"]
                    body_to_send = email_body

                    if personalize and "name" in df.columns:
                        person_name = row["name"]
                        if "Dear Friend" in body_to_send:
                            body_to_send = body_to_send.replace("Dear Friend", f"Dear {person_name}")
                        elif body_to_send.strip().startswith("Dear"):
                            first_line, rest = body_to_send.split("\n", 1)
                            body_to_send = f"Dear {person_name},\n{rest}"
                        else:
                            body_to_send = f"Dear {person_name},\n\n{body_to_send}"

                    result = send_email(
                        sender_email,
                        sender_password,
                        recipient,
                        email_subject,
                        body_to_send,
                        attachments
                    )
                    if result is True:
                        successes.append(recipient)
                    else:
                        failures.append((recipient, result))

                st.success(f"Emails sent successfully to {len(successes)} people.")
                if failures:
                    st.error(f"Failed to send to: {failures}")


# ================== Auth Check ==================
if "authenticated" in st.session_state and st.session_state["authenticated"]:
    event_page()
else:
    st.error("Please login first.")
