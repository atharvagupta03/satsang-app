
import os
import io
import re
import time
from typing import List, Tuple

import streamlit as st
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import google.generativeai as genai

# ---------------------------
# Authentication check (do NOT change)
# ---------------------------
if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    st.warning("Please login from Home page to continue.")
    st.stop()

# ---------------------------
# Gemini / LLM config (use env var)
# ---------------------------
GENAI_API_KEY = "AIzaSyCNtyOWxO9MXLIoZf89d7n6vJEFPrdwoOc"
if not GENAI_API_KEY:
    st.error("GENAI_API_KEY not set in environment.")
    st.stop()

genai.configure(api_key=GENAI_API_KEY)

# Default model and LLM params
LLM_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 1200

def call_gemini_with_config(system_prompt: str, user_prompt: str):
    model = genai.GenerativeModel(LLM_MODEL)
    full_prompt = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}"
    response = model.generate_content(
        full_prompt,
        generation_config={
            "temperature": LLM_TEMPERATURE,
            "max_output_tokens": LLM_MAX_TOKENS,
        },
    )
    return response.text

# ---------------------------
# Utility: PDF chunking + retrieval
# ---------------------------
MAX_PDF_PAGES = 1000
MAX_UPLOAD_MB = 40
MAX_SELECTED_CHARS = 30000
TOP_K_CHUNKS = 6

def read_pdf_bytes(pdf_bytes: bytes) -> List[Tuple[str, int, int]]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    n_pages = min(len(reader.pages), MAX_PDF_PAGES)
    page_texts = []
    for i in range(n_pages):
        try:
            text = reader.pages[i].extract_text() or ""
        except Exception:
            text = ""
        text = re.sub(r'\s+', ' ', text).strip()
        page_texts.append((text, i + 1))
    joined = []
    for t, p in page_texts:
        joined.append(f"[PAGE {p}]\n{t}")
    full = "\n".join(joined)

    chapter_positions = []
    for match in re.finditer(r'(Chapter|CHAPTER|Part|PART|BOOK|Book)\s+([IVX0-9A-Za-z\-]+)', full):
        chapter_positions.append(match.start())

    chunks = []
    if len(chapter_positions) >= 2:
        positions = chapter_positions + [len(full)]
        for i in range(len(chapter_positions)):
            s = positions[i]
            e = positions[i+1]
            snippet = full[s:e].strip()
            pages = re.findall(r'\[PAGE (\d+)\]', snippet)
            if pages:
                start_page = int(pages[0])
                end_page = int(pages[-1])
            else:
                start_page = 1
                end_page = 1
            chunks.append((snippet, start_page, end_page))
    else:
        current_chunk = []
        current_chars = 0
        start_page = None
        for page_text, p in page_texts:
            if start_page is None:
                start_page = p
            block = f"[PAGE {p}]\n{page_text}"
            current_chunk.append(block)
            current_chars += len(block)
            if current_chars > 4000:
                chunk_text = "\n".join(current_chunk).strip()
                chunks.append((chunk_text, start_page, p))
                current_chunk = []
                current_chars = 0
                start_page = None
        if current_chunk:
            chunk_text = "\n".join(current_chunk).strip()
            last_page = page_texts[-1][1] if page_texts else 1
            chunks.append((chunk_text, start_page or 1, last_page))

    final = [(c, s if s else 1, e if e else 1) for (c, s, e) in chunks if c and len(c) > 50]
    return final

def build_tfidf_index(chunks: List[Tuple[str, int, int]]):
    texts = [c[0] for c in chunks]
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1, stop_words='english')
    tfidf = vectorizer.fit_transform(texts)
    return vectorizer, tfidf

def query_top_chunks(query: str, chunks: List[Tuple[str,int,int]], vectorizer, tfidf_matrix, top_k=TOP_K_CHUNKS):
    q_vec = vectorizer.transform([query])
    cosine_similarities = linear_kernel(q_vec, tfidf_matrix).flatten()
    ranked_idx = np.argsort(-cosine_similarities)
    selected = []
    total_chars = 0
    for idx in ranked_idx:
        if cosine_similarities[idx] <= 0:
            break
        chunk_text, sp, ep = chunks[idx]
        if total_chars + len(chunk_text) > MAX_SELECTED_CHARS:
            continue
        selected.append((idx, float(cosine_similarities[idx]), chunk_text, sp, ep))
        total_chars += len(chunk_text)
        if len(selected) >= top_k:
            break
    return selected, cosine_similarities

# ---------------------------
# Helpers: build LLM blog prompt
# ---------------------------
def build_llm_prompt(selected_chunks: List[Tuple[int,float,str,int,int]], topic: str, style: str, blog_length: str):
    sources = []
    for i, (idx, score, text, sp, ep) in enumerate(selected_chunks, start=1):
        header = f"[SOURCE {i}] pages {sp}-{ep} ---"
        sources.append(f"{header}\n{text}")
    sources_text = "\n\n".join(sources)

    system_prompt = (
        "You are a careful writing assistant. You MUST ONLY use the provided SOURCE PASSAGES below to produce the blog."
        " Do NOT use any external knowledge, do not hallucinate, and do not add facts not present in the sources."
        " If the sources do not support an assertion, either omit it or explicitly say 'Not stated in sources'."
    )

    user_instructions = (
        f"Topic / keyword given: '{topic}'.\n"
        f"Writing style: {style or 'neutral, clear, readable'}.\n"
        f"Desired length: {blog_length or 'approx 600-900 words'}.\n\n"
        "INSTRUCTIONS (Must follow exactly):\n"
        "1) Use ONLY the text inside the [SOURCE X] sections below. You may paraphrase but must not introduce new facts.\n"
        "2) Create: (a) A concise headline/title, (b) a 2–3 sentence intro, (c) 3-6 subheaded sections that develop the idea, (d) a concluding paragraph, (e) 3 suggested social-media excerpts (one-liners).\n"
        "3) Every factual claim or direct paraphrase MUST include an inline bracket citation like [SOURCE 1 - pages 12-14].\n"
        "4) If multiple sources support a point, cite each.\n"
        "5) If something cannot be derived from the sources, say 'Not stated in sources'.\n"
        "6) Avoid adding any background not present in the sources. Keep tone consistent with requested style.\n\n"
        "SOURCES:\n"
        + sources_text +
        "\n\nProduce the blog now."
    )

    return system_prompt, user_instructions

# ---------------------------
# Helpers: email sending
# ---------------------------
def send_emails_smtp(smtp_server: str, smtp_port: int, username: str, password: str,
                     sender_name: str, sender_email: str, subject: str,
                     body_html: str, recipients_df: pd.DataFrame, dry_run=False):
    results = {"sent": [], "failed": []}
    if dry_run:
        for _, row in recipients_df.iterrows():
            results["sent"].append({"email": row.get("email"), "status": "dry_run"})
        return results

    server = smtplib.SMTP(smtp_server, smtp_port)
    server.ehlo()
    try:
        server.starttls()
    except Exception:
        pass
    server.login(username, password)

    for _, row in recipients_df.iterrows():
        to_email = row.get("email")
        name = row.get("name", "")
        if not isinstance(to_email, str) or "@" not in to_email:
            results["failed"].append({"email": to_email, "error": "invalid email"})
            continue
        personalized_body = body_html.replace("{name}", name if name else "")
        msg = MIMEMultipart("alternative")
        msg["From"] = f"{sender_name} <{sender_email}>"
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(personalized_body, "html"))
        try:
            server.sendmail(sender_email, to_email, msg.as_string())
            results["sent"].append({"email": to_email})
            time.sleep(0.3)
        except Exception as e:
            results["failed"].append({"email": to_email, "error": str(e)})
    server.quit()
    return results

# ---------------------------
# Page UI
# ---------------------------
st.title("Blog Summarizer (Book → Blog — TF-IDF retrieval, Gemini generation)")
st.markdown("Upload a PDF, enter topic/keyword(s), select retrieval settings, generate a blog from selected passages, edit it, and email to recipients.")

col1, col2 = st.columns([2, 1])

with col1:
    pdf_file = st.file_uploader("Upload book PDF", type=["pdf"], key="blog_pdf_uploader")
    topic = st.text_input("Topic / keyword(s)", value=st.session_state.get("blog_topic", ""), key="blog_topic_input")
    style = st.selectbox("Writing style", ["Neutral and clear", "Conversational", "Formal", "Reflective"], index=0, key="blog_style")
    blog_length = st.selectbox("Desired blog length", ["Short (~400-600 words)", "Medium (~600-900 words)", "Long (~900-1500 words)"], index=1, key="blog_length_select")
    if pdf_file:
        if pdf_file.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"Uploaded file size exceeds {MAX_UPLOAD_MB} MB limit.")
        else:
            raw_bytes = pdf_file.read()
            chunks = read_pdf_bytes(raw_bytes)
            st.success(f"Extracted {len(chunks)} candidate chunks from PDF.")
            if len(chunks) > 0:
                sample_preview = chunks[0][0][:800] + ("..." if len(chunks[0][0]) > 800 else "")
                st.text_area("Sample chunk preview (first chunk)", value=sample_preview, height=200, key="sample_preview_area")
    else:
        chunks = None

with col2:
    top_k = st.number_input("Max chunks to select", min_value=1, max_value=12, value=TOP_K_CHUNKS, key="top_k_input")
    max_chars = st.number_input("Max characters to send to LLM", min_value=5000, max_value=80000, value=MAX_SELECTED_CHARS, step=1000, key="max_chars_input")
    if st.button("Find relevant passages & prepare prompt", disabled=(chunks is None or not topic.strip()), key="find_passages_btn"):
        with st.spinner("Building TF-IDF index..."):
            vectorizer, tfidf_matrix = build_tfidf_index(chunks)
        with st.spinner("Querying top chunks..."):
            selected, similarities = query_top_chunks(topic, chunks, vectorizer, tfidf_matrix, top_k=int(top_k))
        if not selected:
            st.warning("No relevant passages found.")
        else:
            st.success(f"Selected {len(selected)} chunks.")
            sel_info = []
            for i, (idx, score, text, sp, ep) in enumerate(selected, start=1):
                snippet = text[:600] + ("..." if len(text) > 600 else "")
                sel_info.append({"source_id": f"SOURCE {i}", "pages": f"{sp}-{ep}", "score": float(score), "snippet": snippet})
            df_sel = pd.DataFrame(sel_info)
            st.dataframe(df_sel[["source_id", "pages", "score"]], use_container_width=True)

            labeled_selected = []
            for i, (idx, score, text, sp, ep) in enumerate(selected, start=1):
                labeled_selected.append((idx, float(score), text, sp, ep))
            system_prompt, user_prompt = build_llm_prompt(labeled_selected, topic, style, blog_length)
            st.session_state["llm_system"] = system_prompt
            st.session_state["llm_user"] = user_prompt
            st.session_state["selected_chunks"] = labeled_selected

st.markdown("---")
st.header("Generate & Edit Blog")
col_a, col_b = st.columns([2, 1])
with col_a:
    if st.button("Generate blog (Gemini)", disabled=("selected_chunks" not in st.session_state or not st.session_state["selected_chunks"]), key="generate_blog_btn"):
        if "llm_system" not in st.session_state or "llm_user" not in st.session_state:
            st.error("No prompt available.")
        else:
            try:
                with st.spinner("Calling Gemini..."):
                    blog_text = call_gemini_with_config(st.session_state["llm_system"], st.session_state["llm_user"])
                    st.session_state["generated_blog"] = blog_text
                    st.success("Blog generated.")
            except Exception as e:
                st.error(f"Gemini call failed: {e}")

    generated = st.session_state.get("generated_blog", "")
    st.subheader("Generated blog (editable)")
    st.session_state["edited_blog"] = st.text_area("Edit the blog here:", value=generated, height=400, key="edited_blog_area")

with col_b:
    st.subheader("Selected sources")
    if st.session_state.get("selected_chunks"):
        for i, (idx, score, text, sp, ep) in enumerate(st.session_state["selected_chunks"], start=1):
            st.markdown(f"**SOURCE {i}** — pages {sp}-{ep} — relevance {score:.3f}")
            st.write(text[:400] + ("..." if len(text) > 400 else ""))
            st.markdown("---")

st.markdown("---")
st.header("Email: preview & send (optional)")
st.write("Upload CSV with at least an 'email' column. Optional 'name' column. Use `{name}` placeholder in body.")

csv_file = st.file_uploader("Upload recipients CSV", type=["csv"], key="recipients_csv")
sender_email = st.text_input("Sender email (from)", value=os.getenv("SENDER_EMAIL") or "", key="sender_email_input")
sender_name = st.text_input("Sender name", value=os.getenv("SENDER_NAME") or "", key="sender_name_input")
smtp_server = st.text_input("SMTP server", value=os.getenv("SMTP_SERVER") or "smtp.gmail.com", key="smtp_server_input")
smtp_port = st.number_input("SMTP port", value=int(os.getenv("SMTP_PORT") or 587), key="smtp_port_input")
smtp_username = st.text_input("SMTP username", value=os.getenv("SMTP_USERNAME") or sender_email, key="smtp_username_input")
smtp_password = st.text_input("SMTP password / app password", type="password", value=os.getenv("SMTP_PASSWORD") or "", key="smtp_password_input")

email_subject = st.text_input("Email subject", value="A blog I thought you'd like", key="email_subject_input")
email_preview = st.text_area("Email HTML body", value=st.session_state.get("edited_blog", ""), height=300, key="email_preview_area")

col_send1, col_send2 = st.columns([1, 1])
with col_send1:
    if st.button("Preview recipients", disabled=(csv_file is None), key="preview_recipients_btn"):
        try:
            recipients_df = pd.read_csv(csv_file)
            st.session_state["recipients_df"] = recipients_df
            st.dataframe(recipients_df.head(20), use_container_width=True)
            st.success(f"{len(recipients_df)} recipients loaded.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

with col_send2:
    if st.button("Send emails", disabled=("recipients_df" not in st.session_state or not smtp_password or not sender_email), key="send_emails_btn"):
        recipients_df = st.session_state.get("recipients_df")
        if recipients_df is None or "email" not in recipients_df.columns:
            st.error("Recipients CSV missing or no 'email' column.")
        else:
            with st.spinner("Sending emails..."):
                body_html = email_preview.replace("\n", "<br/>")
                results = send_emails_smtp(
                    smtp_server=smtp_server, smtp_port=int(smtp_port),
                    username=smtp_username, password=smtp_password,
                    sender_name=sender_name or sender_email, sender_email=sender_email,
                    subject=email_subject, body_html=body_html,
                    recipients_df=recipients_df, dry_run=False
                )
                st.success(f"Sent: {len(results['sent'])}, Failed: {len(results['failed'])}")
                if results["failed"]:
                    st.write("Failed details:")
                    st.json(results["failed"])



































































































# import streamlit as st
# import os
# import io
# import re
# import time
# from typing import List, Tuple

# import streamlit as st
# import pandas as pd
# import numpy as np
# from PyPDF2 import PdfReader
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import linear_kernel
# import smtplib
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
# import google.generativeai as genai
# from utils import log_action


# if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
#     st.warning("Please login from Home page to continue.")
#     st.stop()

# GENAI_API_KEY = "AIzaSyCNtyOWxO9MXLIoZf89d7n6vJEFPrdwoOc"
# genai.configure(api_key=GENAI_API_KEY)

# # Default model and LLM params (same as your production code)
# LLM_MODEL = "gemini-2.5-flash"
# LLM_TEMPERATURE = 0.2
# LLM_MAX_TOKENS = 1200
# # def newsletter_page():
# #     st.title("📰 Newsletter Verticle")
# #     st.write("Substack and Medium article modules")

# #     if st.button("Substack"):
# #         log_action(st.session_state["username"], "Open", "Newsletter → Substack")
# #         st.info("🔹 Substack module coming soon.")

# #     if st.button("Medium"):
# #         log_action(st.session_state["username"], "Open", "Newsletter → Medium")
# #         st.info("🔹 Medium module coming soon.")

# # if "authenticated" in st.session_state and st.session_state["authenticated"]:
# #     newsletter_page()
# # else:
# #     st.error("Please login first.")


#     st.header("Blog Summarizer (Book → Blog — TF-IDF retrieval, Gemini generation)")
#     st.markdown("Upload a PDF, enter topic/keyword(s), select retrieval settings, generate a blog from selected passages, edit it, and email to recipients.")

#     col1, col2 = st.columns([2, 1])

#     with col1:
#         pdf_file = st.file_uploader("Upload book PDF", type=["pdf"], key="blog_pdf_uploader")
#         topic = st.text_input("Topic / keyword(s)", value=st.session_state.get("blog_topic", ""), key="blog_topic_input")
#         style = st.selectbox("Writing style", ["Neutral and clear", "Conversational", "Formal", "Reflective"], index=0, key="blog_style")
#         blog_length = st.selectbox("Desired blog length", ["Short (~400-600 words)", "Medium (~600-900 words)", "Long (~900-1500 words)"], index=1, key="blog_length_select")
#         if pdf_file:
#             if pdf_file.size > MAX_UPLOAD_MB * 1024 * 1024:
#                 st.error(f"Uploaded file size exceeds {MAX_UPLOAD_MB} MB limit.")
#             else:
#                 raw_bytes = pdf_file.read()
#                 chunks = read_pdf_bytes(raw_bytes)
#                 st.success(f"Extracted {len(chunks)} candidate chunks from PDF.")
#                 if len(chunks) > 0:
#                     sample_preview = chunks[0][0][:800] + ("..." if len(chunks[0][0]) > 800 else "")
#                     st.text_area("Sample chunk preview (first chunk)", value=sample_preview, height=200, key="sample_preview_area")
#         else:
#             chunks = None

#     with col2:
#         top_k = st.number_input("Max chunks to select", min_value=1, max_value=12, value=TOP_K_CHUNKS, key="top_k_input")
#         max_chars = st.number_input("Max characters to send to LLM", min_value=5000, max_value=80000, value=MAX_SELECTED_CHARS, step=1000, key="max_chars_input")
#         if st.button("Find relevant passages & prepare prompt", disabled=(chunks is None or not topic.strip()), key="find_passages_btn"):
#             with st.spinner("Building TF-IDF index..."):
#                 vectorizer, tfidf_matrix = build_tfidf_index(chunks)
#             with st.spinner("Querying top chunks..."):
#                 selected, similarities = query_top_chunks(topic, chunks, vectorizer, tfidf_matrix, top_k=int(top_k))
#             if not selected:
#                 st.warning("No relevant passages found.")
#             else:
#                 st.success(f"Selected {len(selected)} chunks.")
#                 sel_info = []
#                 for i, (idx, score, text, sp, ep) in enumerate(selected, start=1):
#                     snippet = text[:600] + ("..." if len(text) > 600 else "")
#                     sel_info.append({"source_id": f"SOURCE {i}", "pages": f"{sp}-{ep}", "score": float(score), "snippet": snippet})
#                 df_sel = pd.DataFrame(sel_info)
#                 st.dataframe(df_sel[["source_id", "pages", "score"]], use_container_width=True)

#                 labeled_selected = []
#                 for i, (idx, score, text, sp, ep) in enumerate(selected, start=1):
#                     labeled_selected.append((idx, float(score), text, sp, ep))
#                 system_prompt, user_prompt = build_llm_prompt(labeled_selected, topic, style, blog_length)
#                 st.session_state["llm_system"] = system_prompt
#                 st.session_state["llm_user"] = user_prompt
#                 st.session_state["selected_chunks"] = labeled_selected

#     st.markdown("---")
#     st.header("Generate & Edit Blog")
#     col_a, col_b = st.columns([2, 1])
#     with col_a:
#         if st.button("Generate blog (Gemini)", disabled=("selected_chunks" not in st.session_state or not st.session_state["selected_chunks"]), key="generate_blog_btn"):
#             if "llm_system" not in st.session_state or "llm_user" not in st.session_state:
#                 st.error("No prompt available.")
#             else:
#                 try:
#                     with st.spinner("Calling Gemini..."):
#                         blog_text = call_gemini_with_config(st.session_state["llm_system"], st.session_state["llm_user"])
#                         st.session_state["generated_blog"] = blog_text
#                         st.success("Blog generated.")
#                 except Exception as e:
#                     st.error(f"Gemini call failed: {e}")

#         generated = st.session_state.get("generated_blog", "")
#         st.subheader("Generated blog (editable)")
#         st.session_state["edited_blog"] = st.text_area("Edit the blog here:", value=generated, height=400, key="edited_blog_area")

#     with col_b:
#         st.subheader("Selected sources")
#         if st.session_state.get("selected_chunks"):
#             for i, (idx, score, text, sp, ep) in enumerate(st.session_state["selected_chunks"], start=1):
#                 st.markdown(f"**SOURCE {i}** — pages {sp}-{ep} — relevance {score:.3f}")
#                 st.write(text[:400] + ("..." if len(text) > 400 else ""))
#                 st.markdown("---")

#     st.markdown("---")
#     st.header("Email: preview & send (optional)")
#     st.write("Upload CSV with at least an 'email' column. Optional 'name' column. Use `{name}` placeholder in body.")

#     csv_file = st.file_uploader("Upload recipients CSV", type=["csv"], key="recipients_csv")
#     sender_email = st.text_input("Sender email (from)", value=os.getenv("SENDER_EMAIL") or "", key="sender_email_input")
#     sender_name = st.text_input("Sender name", value=os.getenv("SENDER_NAME") or "", key="sender_name_input")
#     smtp_server = st.text_input("SMTP server", value=os.getenv("SMTP_SERVER") or "smtp.gmail.com", key="smtp_server_input")
#     smtp_port = st.number_input("SMTP port", value=int(os.getenv("SMTP_PORT") or 587), key="smtp_port_input")
#     smtp_username = st.text_input("SMTP username", value=os.getenv("SMTP_USERNAME") or sender_email, key="smtp_username_input")
#     smtp_password = st.text_input("SMTP password / app password", type="password", value=os.getenv("SMTP_PASSWORD") or "", key="smtp_password_input")

#     email_subject = st.text_input("Email subject", value="A blog I thought you'd like", key="email_subject_input")
#     email_preview = st.text_area("Email HTML body", value=st.session_state.get("edited_blog", ""), height=300, key="email_preview_area")

#     col_send1, col_send2 = st.columns([1, 1])
#     with col_send1:
#         if st.button("Preview recipients", disabled=(csv_file is None), key="preview_recipients_btn"):
#             try:
#                 recipients_df = pd.read_csv(csv_file)
#                 st.session_state["recipients_df"] = recipients_df
#                 st.dataframe(recipients_df.head(20), use_container_width=True)
#                 st.success(f"{len(recipients_df)} recipients loaded.")
#             except Exception as e:
#                 st.error(f"Failed to read CSV: {e}")

#     with col_send2:
#         if st.button("Send emails", disabled=("recipients_df" not in st.session_state or not smtp_password or not sender_email), key="send_emails_btn"):
#             recipients_df = st.session_state.get("recipients_df")
#             if recipients_df is None or "email" not in recipients_df.columns:
#                 st.error("Recipients CSV missing or no 'email' column.")
#             else:
#                 with st.spinner("Sending emails..."):
#                     body_html = email_preview.replace("\n", "<br/>")
#                     results = send_emails_smtp(
#                         smtp_server=smtp_server, smtp_port=int(smtp_port),
#                         username=smtp_username, password=smtp_password,
#                         sender_name=sender_name or sender_email, sender_email=sender_email,
#                         subject=email_subject, body_html=body_html,
#                         recipients_df=recipients_df, dry_run=False
#                     )
#                     st.success(f"Sent: {len(results['sent'])}, Failed: {len(results['failed'])}")
#                     if results["failed"]:
#                         st.write("Failed details:")
#                         st.json(results["failed"])
