# pages/1_Rewriting.py
import os
import io
import re
import time
from typing import List, Tuple

import streamlit as st
# from dotenv import load_dotenv
import pandas as pd
import numpy as np
# from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import google.generativeai as genai

# ---------------------------
# Authentication check 
# ---------------------------
if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    st.warning("Please login from Home page to continue.")
    st.stop()

# ---------------------------
# Gemini config
# ---------------------------

GENAI_API_KEY = st.secrets("GENAI_API_KEY")
genai.configure(api_key=GENAI_API_KEY)

# Default model and LLM params (same as your production code)
LLM_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 1200

def call_gemini_with_config(system_prompt: str, user_prompt: str):
    """
    Call Gemini with system+user prompt, using generation_config controlling temp/tokens.
    Returns response text or raises exception.
    """
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

def call_gemini_simple(prompt: str):
    """Simple single-prompt call (for summarize/rephrase/translate flows)."""
    model = genai.GenerativeModel(LLM_MODEL)
    response = model.generate_content(prompt)
    return response.text

# ---------------------------
# Utility: PDF chunking + retrieval (exact production logic)
# ---------------------------
# MAX_PDF_PAGES = 1000
# MAX_UPLOAD_MB = 40
# MAX_SELECTED_CHARS = 30000
# TOP_K_CHUNKS = 6

# def read_pdf_bytes(pdf_bytes: bytes) -> List[Tuple[str, int, int]]:
#     reader = PdfReader(io.BytesIO(pdf_bytes))
#     n_pages = min(len(reader.pages), MAX_PDF_PAGES)
#     page_texts = []
#     for i in range(n_pages):
#         try:
#             text = reader.pages[i].extract_text() or ""
#         except Exception:
#             text = ""
#         text = re.sub(r'\s+', ' ', text).strip()
#         page_texts.append((text, i + 1))
#     joined = []
#     for t, p in page_texts:
#         joined.append(f"[PAGE {p}]\n{t}")
#     full = "\n".join(joined)

#     chapter_positions = []
#     for match in re.finditer(r'(Chapter|CHAPTER|Part|PART|BOOK|Book)\s+([IVX0-9A-Za-z\-]+)', full):
#         chapter_positions.append(match.start())

#     chunks = []
#     if len(chapter_positions) >= 2:
#         positions = chapter_positions + [len(full)]
#         for i in range(len(chapter_positions)):
#             s = positions[i]
#             e = positions[i+1]
#             snippet = full[s:e].strip()
#             pages = re.findall(r'\[PAGE (\d+)\]', snippet)
#             if pages:
#                 start_page = int(pages[0])
#                 end_page = int(pages[-1])
#             else:
#                 start_page = 1
#                 end_page = 1
#             chunks.append((snippet, start_page, end_page))
#     else:
#         current_chunk = []
#         current_chars = 0
#         start_page = None
#         for page_text, p in page_texts:
#             if start_page is None:
#                 start_page = p
#             block = f"[PAGE {p}]\n{page_text}"
#             current_chunk.append(block)
#             current_chars += len(block)
#             if current_chars > 4000:
#                 chunk_text = "\n".join(current_chunk).strip()
#                 chunks.append((chunk_text, start_page, p))
#                 current_chunk = []
#                 current_chars = 0
#                 start_page = None
#         if current_chunk:
#             chunk_text = "\n".join(current_chunk).strip()
#             last_page = page_texts[-1][1] if page_texts else 1
#             chunks.append((chunk_text, start_page or 1, last_page))

#     final = [(c, s if s else 1, e if e else 1) for (c, s, e) in chunks if c and len(c) > 50]
#     return final

# def build_tfidf_index(chunks: List[Tuple[str, int, int]]):
#     texts = [c[0] for c in chunks]
#     vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1, stop_words='english')
#     tfidf = vectorizer.fit_transform(texts)
#     return vectorizer, tfidf

# def query_top_chunks(query: str, chunks: List[Tuple[str,int,int]], vectorizer, tfidf_matrix, top_k=TOP_K_CHUNKS):
#     q_vec = vectorizer.transform([query])
#     cosine_similarities = linear_kernel(q_vec, tfidf_matrix).flatten()
#     ranked_idx = np.argsort(-cosine_similarities)
#     selected = []
#     total_chars = 0
#     for idx in ranked_idx:
#         if cosine_similarities[idx] <= 0:
#             break
#         chunk_text, sp, ep = chunks[idx]
#         if total_chars + len(chunk_text) > MAX_SELECTED_CHARS:
#             continue
#         selected.append((idx, float(cosine_similarities[idx]), chunk_text, sp, ep))
#         total_chars += len(chunk_text)
#         if len(selected) >= top_k:
#             break
#     return selected, cosine_similarities

# ---------------------------
# Helpers: build LLM blog prompt (exact production)
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
        "\n\n"
    )

    return system_prompt, user_instructions

# ---------------------------
# Helpers: email sending (production-ready)
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
# Page UI: tabs for Summarize / Rephrase / Translate / Blog Summarizer
# ---------------------------

st.title("Rewriting Module")

tabs = st.tabs(["Summarize", "Rephrase (guided)", "Translate (EN ↔ HI)", "prompt input"])

# --- SESSION KEYS: ensure persistence, unique keys used in UI widgets
if "summarize_input" not in st.session_state:
    st.session_state["summarize_input"] = ""
if "summary" not in st.session_state:
    st.session_state["summary"] = ""
if "rephrase_input" not in st.session_state:
    st.session_state["rephrase_input"] = ""
if "rephrase_instruction" not in st.session_state:
    st.session_state["rephrase_instruction"] = ""
if "rephrased" not in st.session_state:
    st.session_state["rephrased"] = ""
if "translate_input" not in st.session_state:
    st.session_state["translate_input"] = ""
if "translate_dir_idx" not in st.session_state:
    st.session_state["translate_dir_idx"] = 0
if "translation" not in st.session_state:
    st.session_state["translation"] = ""
if "blog_topic" not in st.session_state:
    st.session_state["blog_topic"] = ""
if "selected_chunks" not in st.session_state:
    st.session_state["selected_chunks"] = None
if "llm_system" not in st.session_state:
    st.session_state["llm_system"] = ""
if "llm_user" not in st.session_state:
    st.session_state["llm_user"] = ""
if "generated_blog" not in st.session_state:
    st.session_state["generated_blog"] = ""
if "edited_blog" not in st.session_state:
    st.session_state["edited_blog"] = ""

# ---------------------------
# Tab: Summarize
# ---------------------------
with tabs[0]:
    st.header("Summarize Text")
    st.session_state["summarize_input"] = st.text_area(
        "Enter text to summarize:",
        value=st.session_state["summarize_input"],
        height=200,
        key="summarize_input_area"
    )
    if st.button("Generate Summary", key="summarize_go"):
        if st.session_state["summarize_input"].strip():
            prompt = f"You are the summarizer for Ramashram Satsang, Mathura; read the provided spiritual text (discourse, reflection, meditation guide, etc.) and condense it into a gentle, uplifting, and very humanized summary that reflects light, love, peace, and meditation; capture essential meaning, central themes, and spiritual insights without adding anything new; keep it brief (long text: 2–3 paragraphs, medium/short: 1 paragraph); write in a calm, devotional, inspiring tone, as if personally guiding someone in meditation; ensure the language feels warm, natural, and human, never mechanical; retain spiritual terms (Dhyan, Satsang, Transcendent Light) in transliteration when needed; output plain text only, polished, leaving the reader peaceful, connected, and inspired. :\n\n{st.session_state['summarize_input']}"
            out = call_gemini_simple(prompt)
            st.session_state["summary"] = out
        else:
            st.warning("Please enter text to summarize.")

    if st.session_state["summary"]:
        st.subheader("Summary (editable)")
        st.session_state["summary"] = st.text_area("Editable Summary",
                                                  value=st.session_state["summary"],
                                                  height=200,
                                                  key="summary_edit_area")

# ---------------------------
# Tab: Rephrase (guided)
# ---------------------------
with tabs[1]:
    st.header("Rephrase (Human-guided)")
    st.session_state["rephrase_input"] = st.text_area(
        "Enter text to rephrase:",
        value=st.session_state["rephrase_input"],
        height=200,
        key="rephrase_input_area"
    )
    st.session_state["rephrase_instruction"] = st.text_area(
        "Describe how it should be (tone, clarity, meaning to emphasize, audience etc.):",
        value=st.session_state["rephrase_instruction"],
        height=140,
        key="rephrase_instruction_area"
    )
    if st.button("Generate Rephrase", key="rephrase_go"):
        if st.session_state["rephrase_input"].strip() and st.session_state["rephrase_instruction"].strip():
            prompt = (
                f"You are the rephraser for Ramashram Satsang, Mathura; your task is to take the provided text and, with guidance from the user on how they want it to sound (tone, style, emphasis), rephrase it into a very humanized, natural, and devotional form that reflects light, love, peace, and meditation; preserve the original meaning and spiritual depth, but adjust expression to align with user’s intent; write in warm, inspiring, and human language, retaining spiritual terms (Dhyan, Satsang, Transcendent Light) in transliteration when needed; output plain polished text that feels authentic and heartfelt. \n\n"
                f"{st.session_state['rephrase_instruction']}\n\n"
                f"Original Text:\n{st.session_state['rephrase_input']}\n\n"
                f"Provide a polished, faithful rephrasing that preserves meaning and follows the user's instruction."
            )
            out = call_gemini_simple(prompt)
            st.session_state["rephrased"] = out
        else:
            st.warning("Please provide both original text and user instruction.")

    if st.session_state["rephrased"]:
        st.subheader("Rephrased Text (editable)")
        st.session_state["rephrased"] = st.text_area("Editable Rephrase",
                                                     value=st.session_state["rephrased"],
                                                     height=220,
                                                     key="rephrase_edit_area")

# ---------------------------
# Tab: Translate (EN <-> HI)
# ---------------------------
with tabs[2]:
    st.header("Translate (English ↔ Hindi)")
    st.session_state["translate_input"] = st.text_area(
        "Enter text to translate:",
        value=st.session_state["translate_input"],
        height=200,
        key="translate_input_area"
    )
    direction = st.radio("Direction:", ["English → Hindi", "Hindi → English"], index=st.session_state["translate_dir_idx"], key="translate_dir_radio")
    if st.button("Translate Now", key="translate_go"):
        if st.session_state["translate_input"].strip():
            if direction == "English → Hindi":
                prompt = f"Translate the following English text to Hindi with the following instructions -  You are the translator for Ramashram Satsang, Mathura; your task is to translate the provided text faithfully into the target language while ensuring the meaning, spiritual values, and essence are never lost; preserve the devotional depth, light, love, peace, and meditation focus of the content; use very humanized, natural, and inspiring language, never mechanical; retain spiritual terms (Dhyan, Satsang, Transcendent Light) in transliteration when appropriate; output plain polished text that feels authentic, warm, and spiritually alive. \n\n{st.session_state['translate_input']}"
            else:
                prompt = f"Translate the following Hindi text to English with the following instructions - You are the translator for Ramashram Satsang, Mathura; your task is to translate the provided text faithfully into the target language while ensuring the meaning, spiritual values, and essence are never lost; preserve the devotional depth, light, love, peace, and meditation focus of the content; use very humanized, natural, and inspiring language, never mechanical; retain spiritual terms (Dhyan, Satsang, Transcendent Light) in transliteration when appropriate; output plain polished text that feels authentic, warm, and spiritually alive. :\n\n{st.session_state['translate_input']}"
            out = call_gemini_simple(prompt)
            st.session_state["translation"] = out
            st.session_state["translate_dir_idx"] = 0 if direction == "English → Hindi" else 1
        else:
            st.warning("Please enter text to translate.")

    if st.session_state["translation"]:
        st.subheader("Translated Text (editable)")
        st.session_state["translation"] = st.text_area("Editable Translation",
                                                       value=st.session_state["translation"],
                                                       height=200,
                                                       key="translation_edit_area")

# ---------------------------
# Tab: Prompt Input (freeform)
# ---------------------------
with tabs[3]:
    st.header("Prompt Input")

    # Ensure separate state keys for this tab
    if "freeform_prompt" not in st.session_state:
        st.session_state["freeform_prompt"] = ""
    if "freeform_output" not in st.session_state:
        st.session_state["freeform_output"] = ""

    # Text area for user prompt
    st.session_state["freeform_prompt"] = st.text_area(
        "Enter your prompt for Gemini:",
        value=st.session_state["freeform_prompt"],
        height=200,
        key="freeform_prompt_area"
    )

    # Generate button
    if st.button("Generate Response", key="freeform_go"):
        if st.session_state["freeform_prompt"].strip():
            try:
                out = call_gemini_simple(st.session_state["freeform_prompt"])
                st.session_state["freeform_output"] = out
            except Exception as e:
                st.error(f"Error generating response: {e}")
        else:
            st.warning("Please enter a prompt.")

    # Editable output box
    if st.session_state["freeform_output"]:
        st.subheader("Generated Response (editable)")
        st.session_state["freeform_output"] = st.text_area(
            "Editable Response",
            value=st.session_state["freeform_output"],
            height=300,
            key="freeform_output_area"
        )















# ---------------------------
# Tab: Blog Summarizer (production code integrated)
# ---------------------------
# with tabs[3]:
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





















































































































# # # # import streamlit as st
# # # # from utils import log_action

# # # # def rewriting_page():
# # # #     st.title("📝 Rewriting Verticle")
# # # #     st.write("Choose an option: Summarize, Translate, Rephrase.")

# # # #     if st.button("Summarize"):
# # # #         log_action(st.session_state["username"], "Open", "Rewriting → Summarize")
# # # #         st.info("🔹 Summarize module coming soon.")

# # # #     if st.button("Translate"):
# # # #         log_action(st.session_state["username"], "Open", "Rewriting → Translate")
# # # #         st.info("🔹 Translate module coming soon.")

# # # #     if st.button("Rephrase"):
# # # #         log_action(st.session_state["username"], "Open", "Rewriting → Rephrase")
# # # #         st.info("🔹 Rephrase module coming soon.")

# # # # if "authenticated" in st.session_state and st.session_state["authenticated"]:
# # # #     rewriting_page()
# # # # else:
# # # #     st.error("Please login first.")


# # # import streamlit as st
# # # import google.generativeai as genai

# # # # 🔐 Authentication check
# # # if "authenticated" not in st.session_state or not st.session_state.authenticated:
# # #     st.warning("Please login from Home page to continue.")
# # #     st.stop()

# # # # 🔑 Gemini Setup
# # # genai.configure(api_key="AIzaSyDEM9OWpU6xhoTuH2oZqCkE36hGJFBYOyc")
# # # model = genai.GenerativeModel("gemini-2.5-flash")

# # # st.title("Rewriting Module")

# # # # Initialize session state
# # # for key in ["rewrite_input", "rewrite_output", "rewrite_mode", "translate_lang"]:
# # #     if key not in st.session_state:
# # #         st.session_state[key] = ""

# # # # Mode selection
# # # mode = st.radio(
# # #     "Choose Rewriting Task",
# # #     ["Summarize", "Translate", "Rephrase"],
# # #     key="rewrite_mode"
# # # )

# # # # Input text area
# # # st.session_state.rewrite_input = st.text_area(
# # #     "Enter your text:",
# # #     value=st.session_state.rewrite_input,
# # #     height=200
# # # )

# # # # Extra: Language selection if Translate mode
# # # if mode == "Translate":
# # #     st.session_state.translate_lang = st.selectbox(
# # #         "Select target language",
# # #         ["Hindi", "French", "Spanish", "German", "English"],
# # #         index=0
# # #     )

# # # # Generate action
# # # if st.button("Generate"):
# # #     if st.session_state.rewrite_input.strip():
# # #         if mode == "Summarize":
# # #             prompt = f"Summarize this text clearly:\n\n{st.session_state.rewrite_input}"
# # #         elif mode == "Translate":
# # #             prompt = f"Translate this text into {st.session_state.translate_lang}:\n\n{st.session_state.rewrite_input}"
# # #         elif mode == "Rephrase":
# # #             prompt = f"Rephrase this text in a professional tone:\n\n{st.session_state.rewrite_input}"

# # #         try:
# # #             response = model.generate_content(prompt)
# # #             st.session_state.rewrite_output = response.text.strip()
# # #         except Exception as e:
# # #             st.error(f"Error from Gemini API: {e}")
# # #     else:
# # #         st.warning("Please enter some text before generating.")

# # # # Editable output
# # # if st.session_state.rewrite_output:
# # #     st.subheader("Editable Output")
# # #     st.session_state.rewrite_output = st.text_area(
# # #         "Generated Text (editable):",
# # #         value=st.session_state.rewrite_output,
# # #         height=200
# # #     )



# # import streamlit as st
# # import google.generativeai as genai
# # import os
# # from datetime import datetime

# # # ================== CONFIG ==================
# # genai.configure(api_key="AIzaSyCNtyOWxO9MXLIoZf89d7n6vJEFPrdwoOc")
# # model = genai.GenerativeModel("gemini-2.0-flash")

# # LOG_FILE = "logs/activity.log"

# # # ================== LOGGING ==================
# # def log_action(username, action, details=""):
# #     with open(LOG_FILE, "a") as f:
# #         f.write(f"{datetime.now()} | {username} | {action} | {details}\n")

# # # ================== PAGE ==================
# # st.title("📝 Rewriting Module")

# # # Require login info from session_state
# # if "username" not in st.session_state:
# #     st.error("Please login first from Home page.")
# #     st.stop()

# # username = st.session_state["username"]

# # # Initialize session_state for persistence
# # if "rewriting" not in st.session_state:
# #     st.session_state["rewriting"] = {
# #         "mode": "Summarize",
# #         "input_text": "",
# #         "output_text": "",
# #         "translation_direction": "English → Hindi",
# #     }

# # # ================== MODE SELECTION ==================
# # mode = st.radio(
# #     "Choose an operation:",
# #     ["Summarize", "Translate", "Rephrase"],
# #     index=["Summarize", "Translate", "Rephrase"].index(st.session_state["rewriting"]["mode"]),
# # )

# # st.session_state["rewriting"]["mode"] = mode

# # # ================== INPUT ==================
# # input_text = st.text_area(
# #     "Enter your text here:",
# #     value=st.session_state["rewriting"]["input_text"],
# #     height=200,
# # )
# # st.session_state["rewriting"]["input_text"] = input_text

# # # ================== TRANSLATION DIRECTION ==================
# # translation_direction = None
# # if mode == "Translate":
# #     translation_direction = st.radio(
# #         "Select Translation Direction:",
# #         ["English → Hindi", "Hindi → English"],
# #         index=["English → Hindi", "Hindi → English"].index(
# #             st.session_state["rewriting"]["translation_direction"]
# #         ),
# #     )
# #     st.session_state["rewriting"]["translation_direction"] = translation_direction

# # # ================== GENERATE BUTTON ==================
# # if st.button("Generate"):
# #     if not input_text.strip():
# #         st.warning("Please enter some text first.")
# #     else:
# #         if mode == "Summarize":
# #             prompt = f" You are the summarizer for Ramashram Satsang, Mathura; read the provided spiritual text (discourse, reflection, meditation guide, etc.) and condense it into a gentle, uplifting summary that reflects light, love, peace, and meditation; capture essential meaning, central themes, and spiritual insights without adding anything new; keep it brief (long text: 2–3 paragraphs, medium/short: 1 paragraph); use a calm, devotional, inspiring tone as if guiding someone in meditation; retain spiritual terms (Dhyan, Satsang, Transcendent Light) in transliteration when needed; output plain text only, polished, leaving the reader peaceful, connected, and inspired. \n\n{input_text}"

# #         elif mode == "Rephrase":
# #             prompt = f"Rephrase the following text to make it more formal, polished, and fluent:\n\n{input_text}"

# #         elif mode == "Translate":
# #             if translation_direction == "English → Hindi":
# #                 prompt = f"Translate the following English text to Hindi:\n\n{input_text}"
# #             else:
# #                 prompt = f"Translate the following Hindi text to English:\n\n{input_text}"

# #         try:
# #             response = model.generate_content(prompt)
# #             output_text = response.text.strip()
# #             st.session_state["rewriting"]["output_text"] = output_text
# #             log_action(username, f"{mode} Generated", f"Direction={translation_direction if mode=='Translate' else ''}")
# #         except Exception as e:
# #             st.error(f"Error while generating: {e}")

# # # ================== OUTPUT ==================
# # if st.session_state["rewriting"]["output_text"]:
# #     st.subheader("Editable Output")
# #     st.session_state["rewriting"]["output_text"] = st.text_area(
# #         "Output",
# #         value=st.session_state["rewriting"]["output_text"],
# #         height=250,
# #     )





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

# # =======================
# # Streamlit Page Config (must be first Streamlit call)
# # =======================
# st.set_page_config(page_title="Vertical 1: Content Tools", layout="wide")

# # =======================
# # Gemini API Setup
# # =======================
# GENAI_API_KEY = os.getenv("GENAI_API_KEY")
# genai.configure(api_key=GENAI_API_KEY)
# model = genai.GenerativeModel("gemini-1.5-flash")

# LLM_TEMPERATURE = 0.2
# LLM_MAX_TOKENS = 1200

# # =======================
# # Authentication
# # =======================
# AUTH_USERNAME = os.getenv("APP_USERNAME", "admin")
# AUTH_PASSWORD = os.getenv("APP_PASSWORD", "rsm@123")

# if "authenticated" not in st.session_state:
#     st.session_state["authenticated"] = False

# if not st.session_state["authenticated"]:
#     st.title("🔒 Login Required")
#     username = st.text_input("Username", key="login_user")
#     password = st.text_input("Password", type="password", key="login_pass")
#     if st.button("Login", key="login_btn"):
#         if username == AUTH_USERNAME and password == AUTH_PASSWORD:
#             st.session_state["authenticated"] = True
#             st.success("Login successful!")
#             st.rerun()
#         else:
#             st.error("Invalid credentials")
#     st.stop()

# # =======================
# # Main App Layout
# # =======================
# st.title("Vertical 1 — Content Processing Suite")

# tab1, tab2, tab3, tab4 = st.tabs([
#     "Summarize",
#     "Rephrase (with human input)",
#     "Translate (EN ↔ HI)",
#     "Blog Summarizer"
# ])

# # =======================
# # Helper: LLM call wrapper
# # =======================
# def call_llm(prompt: str):
#     try:
#         response = model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         return f"Error: {str(e)}"

# # =======================
# # TAB 1: Summarizer
# # =======================
# with tab1:
#     st.header("Summarize Text")
#     text_input = st.text_area(
#         "Enter text to summarize:",
#         value=st.session_state.get("summarize_input", ""),
#         height=200,
#         key="summarize_input_area"
#     )
#     if st.button("Summarize", key="summarize_btn"):
#         if text_input.strip():
#             st.session_state["summarize_input"] = text_input
#             prompt = f"Summarize this text clearly and concisely:\n\n{text_input}"
#             summary = call_llm(prompt)
#             st.session_state["summary"] = summary
#         else:
#             st.warning("Please enter text to summarize.")

#     if "summary" in st.session_state:
#         st.subheader("Summary")
#         st.text_area(
#             "Editable Summary",
#             value=st.session_state["summary"],
#             height=200,
#             key="summary_edit"
#         )

# # =======================
# # TAB 2: Rephrase (Human-guided)
# # =======================
# with tab2:
#     st.header("Rephrase Text (Guided)")
#     text_input = st.text_area(
#         "Enter text to rephrase:",
#         value=st.session_state.get("rephrase_input", ""),
#         height=200,
#         key="rephrase_input_area"
#     )
#     human_instruction = st.text_area(
#         "What do you want it to look like?",
#         value=st.session_state.get("rephrase_instruction", ""),
#         height=120,
#         key="rephrase_instruction_area"
#     )
#     if st.button("Rephrase", key="rephrase_btn"):
#         if text_input.strip() and human_instruction.strip():
#             st.session_state["rephrase_input"] = text_input
#             st.session_state["rephrase_instruction"] = human_instruction
#             prompt = (
#                 f"Rephrase the following text based on the user's intent:\n\n"
#                 f"Original Text:\n{text_input}\n\n"
#                 f"User Instruction:\n{human_instruction}\n\n"
#                 f"Now provide the rephrased version."
#             )
#             rephrased = call_llm(prompt)
#             st.session_state["rephrased"] = rephrased
#         else:
#             st.warning("Please provide both text and instructions.")

#     if "rephrased" in st.session_state:
#         st.subheader("Rephrased Text")
#         st.text_area(
#             "Editable Rephrased",
#             value=st.session_state["rephrased"],
#             height=200,
#             key="rephrase_edit"
#         )

# # =======================
# # TAB 3: Translate (EN ↔ HI only)
# # =======================
# with tab3:
#     st.header("Translate Text (English ↔ Hindi)")
#     text_input = st.text_area(
#         "Enter text to translate:",
#         value=st.session_state.get("translate_input", ""),
#         height=200,
#         key="translate_input_area"
#     )
#     direction = st.radio(
#         "Choose translation direction:",
#         ["English → Hindi", "Hindi → English"],
#         index=st.session_state.get("translate_dir_idx", 0),
#         key="translate_dir"
#     )

#     if st.button("Translate", key="translate_btn"):
#         if text_input.strip():
#             st.session_state["translate_input"] = text_input
#             st.session_state["translate_dir_idx"] = 0 if direction == "English → Hindi" else 1
#             if direction == "English → Hindi":
#                 prompt = f"Translate the following English text to Hindi:\n\n{text_input}"
#             else:
#                 prompt = f"Translate the following Hindi text to English:\n\n{text_input}"
#             translation = call_llm(prompt)
#             st.session_state["translation"] = translation
#         else:
#             st.warning("Please enter text to translate.")

#     if "translation" in st.session_state:
#         st.subheader("Translated Text")
#         st.text_area(
#             "Editable Translation",
#             value=st.session_state["translation"],
#             height=200,
#             key="translate_edit"
#         )

# # =======================
# # TAB 4: Blog Summarizer (Book → Blog)
# # =======================
# with tab4:
#     MAX_PDF_PAGES = 1000
#     MAX_UPLOAD_MB = 40
#     MAX_SELECTED_CHARS = 30000
#     TOP_K_CHUNKS = 6

#     def read_pdf_bytes(pdf_bytes: bytes) -> List[Tuple[str, int, int]]:
#         reader = PdfReader(io.BytesIO(pdf_bytes))
#         n_pages = min(len(reader.pages), MAX_PDF_PAGES)
#         page_texts = []
#         for i in range(n_pages):
#             try:
#                 text = reader.pages[i].extract_text() or ""
#             except Exception:
#                 text = ""
#             text = re.sub(r"\s+", " ", text).strip()
#             page_texts.append((text, i + 1))
#         return [(t, p, p) for t, p in page_texts if t]

#     def build_tfidf_index(chunks):
#         texts = [c[0] for c in chunks]
#         vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
#         tfidf = vectorizer.fit_transform(texts)
#         return vectorizer, tfidf

#     def query_top_chunks(query, chunks, vectorizer, tfidf_matrix, top_k=TOP_K_CHUNKS):
#         q_vec = vectorizer.transform([query])
#         cosine_similarities = linear_kernel(q_vec, tfidf_matrix).flatten()
#         ranked_idx = np.argsort(-cosine_similarities)
#         selected = []
#         total_chars = 0
#         for idx in ranked_idx:
#             if cosine_similarities[idx] <= 0:
#                 break
#             chunk_text, sp, ep = chunks[idx]
#             if total_chars + len(chunk_text) > MAX_SELECTED_CHARS:
#                 continue
#             selected.append((idx, cosine_similarities[idx], chunk_text, sp, ep))
#             total_chars += len(chunk_text)
#             if len(selected) >= top_k:
#                 break
#         return selected

#     def send_emails_smtp(smtp_server, smtp_port, username, password,
#                          sender_name, sender_email, subject,
#                          body_html, recipients_df, dry_run=False):
#         results = {"sent": [], "failed": []}
#         if dry_run:
#             for _, row in recipients_df.iterrows():
#                 results["sent"].append({"email": row.get("email"), "status": "dry_run"})
#             return results

#         server = smtplib.SMTP(smtp_server, smtp_port)
#         server.ehlo()
#         try:
#             server.starttls()
#         except Exception:
#             pass
#         server.login(username, password)

#         for _, row in recipients_df.iterrows():
#             to_email = row.get("email")
#             name = row.get("name", "")
#             if not isinstance(to_email, str) or "@" not in to_email:
#                 results["failed"].append({"email": to_email, "error": "invalid email"})
#                 continue
#             personalized_body = body_html.replace("{name}", name if name else "")
#             msg = MIMEMultipart("alternative")
#             msg["From"] = f"{sender_name} <{sender_email}>"
#             msg["To"] = to_email
#             msg["Subject"] = subject
#             msg.attach(MIMEText(personalized_body, "html"))
#             try:
#                 server.sendmail(sender_email, to_email, msg.as_string())
#                 results["sent"].append({"email": to_email})
#                 time.sleep(0.3)
#             except Exception as e:
#                 results["failed"].append({"email": to_email, "error": str(e)})
#         server.quit()
#         return results

#     st.header("Blog Summarizer (Book → Blog)")

#     pdf_file = st.file_uploader("Upload Book PDF", type=["pdf"], key="blog_pdf")
#     topic = st.text_input("Topic / keyword(s)", value=st.session_state.get("blog_topic", ""), key="blog_topic_input")

#     if pdf_file:
#         if pdf_file.size > MAX_UPLOAD_MB * 1024 * 1024:
#             st.error(f"File too large (> {MAX_UPLOAD_MB} MB).")
#         else:
#             raw_bytes = pdf_file.read()
#             chunks = read_pdf_bytes(raw_bytes)
#             st.success(f"Extracted {len(chunks)} chunks.")
#             if topic.strip() and st.button("Generate Blog", key="blog_btn"):
#                 st.session_state["blog_topic"] = topic
#                 vec, tfidf = build_tfidf_index(chunks)
#                 selected = query_top_chunks(topic, chunks, vec, tfidf)
#                 sources_text = "\n\n".join([s[2] for s in selected])
#                 blog_prompt = (
#                     f"You are a writing assistant. Using ONLY the following sources, "
#                     f"write a blog on '{topic}':\n\n{sources_text}"
#                 )
#                 blog_text = call_llm(blog_prompt)
#                 st.session_state["blog_text"] = blog_text

#     if "blog_text" in st.session_state:
#         st.subheader("Generated Blog (Editable)")
#         st.text_area("Edit Blog", value=st.session_state["blog_text"], height=400, key="blog_edit")

#         st.subheader("Email Blog to Recipients")
#         csv_file = st.file_uploader("Upload recipients CSV (columns: email, name)", type=["csv"], key="recipients_csv")
#         sender_email = st.text_input("Sender email (from)", value=os.getenv("SENDER_EMAIL") or "", key="sender_email")
#         sender_name = st.text_input("Sender name", value=os.getenv("SENDER_NAME") or "", key="sender_name")
#         smtp_server = st.text_input("SMTP server", value=os.getenv("SMTP_SERVER") or "smtp.gmail.com", key="smtp_server")
#         smtp_port = st.number_input("SMTP port", value=int(os.getenv("SMTP_PORT") or 587), key="smtp_port")
#         smtp_username = st.text_input("SMTP username", value=os.getenv("SMTP_USERNAME") or sender_email, key="smtp_username")
#         smtp_password = st.text_input("SMTP password / app password", type="password", value=os.getenv("SMTP_PASSWORD") or "", key="smtp_password")

#         email_subject = st.text_input("Email subject", value="A blog I thought you'd like", key="email_subject")
#         email_preview = st.text_area("Email HTML body", value=st.session_state.get("blog_edit", ""), height=300, key="email_preview")

#         if st.button("Send emails", disabled=(csv_file is None or not smtp_password or not sender_email), key="send_emails_btn"):
#             try:
#                 recipients_df = pd.read_csv(csv_file)
#                 if "email" not in recipients_df.columns:
#                     st.error("CSV must contain 'email' column.")
#                 else:
#                     with st.spinner("Sending emails..."):
#                         body_html = email_preview.replace("\n", "<br/>")
#                         results = send_emails_smtp(
#                             smtp_server=smtp_server, smtp_port=int(smtp_port),
#                             username=smtp_username, password=smtp_password,
#                             sender_name=sender_name or sender_email, sender_email=sender_email,
#                             subject=email_subject, body_html=body_html,
#                             recipients_df=recipients_df, dry_run=False
#                         )
#                         st.success(f"Sent: {len(results['sent'])}, Failed: {len(results['failed'])}")
#                         if results["failed"]:
#                             st.write("Failed details:")
#                             st.json(results["failed"])
#             except Exception as e:
#                 st.error(f"Failed to send: {e}")
