"""Interface Streamlit de l'agent de support client (Phase 5).

Lancement :  streamlit run app.py

Rappel : cette application est une démonstration avec des données fictives.
Elle ne contacte JAMAIS un vrai client : « approuver » enregistre seulement
le brouillon dans la session.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from src.analyzer import analyze_message
from src.config import load_settings
from src.handoff import format_handoff
from src.knowledge_base import load_knowledge_base
from src.llm_client import create_llm_client
from src.ui_helpers import (
    MAX_MESSAGE_LENGTH,
    SAMPLE_MESSAGES,
    draft_warnings,
    empty_state_html,
    masthead_html,
    relevance_bar_html,
    ticket_html,
)

st.set_page_config(page_title="Customer support desk", page_icon="💬", layout="centered")

STYLE_PATH = Path(__file__).resolve().parent / "assets" / "style.css"


def inject_style() -> None:
    """Charge la feuille de style (si elle manque, l'application reste utilisable)."""
    try:
        st.html(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>")
    except OSError:
        pass


# --- Ressources partagées (chargées une seule fois) ---------------------------
@st.cache_resource
def get_settings():
    return load_settings()


@st.cache_resource
def get_llm():
    return create_llm_client(get_settings())


@st.cache_resource
def get_kb():
    return load_knowledge_base()


# --- État de la session --------------------------------------------------------
def init_state() -> None:
    defaults = {
        "history": [],  # messages du client dans la conversation en cours
        "result": None,  # dernière analyse
        "analysis_id": 0,  # change à chaque analyse (réinitialise le brouillon)
        "approved": [],  # brouillons approuvés (jamais envoyés)
        "approved_ids": set(),
        "message_input": "",
        "example": "—",
        "toast": None,  # message à afficher après le rechargement
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def load_example() -> None:
    """Copie l'exemple choisi dans la zone de saisie."""
    if st.session_state.example != "—":
        st.session_state.message_input = st.session_state.example


def new_conversation() -> None:
    st.session_state.history = []
    st.session_state.result = None
    st.session_state.message_input = ""
    st.session_state.example = "—"


def approve_draft() -> None:
    """Enregistre le brouillon (éventuellement modifié) comme approuvé."""
    analysis_id = st.session_state.analysis_id
    result = st.session_state.result
    if result is None or analysis_id in st.session_state.approved_ids:
        return
    st.session_state.approved.append(
        {
            "customer_message": st.session_state.history[-1],
            "reply": st.session_state[f"draft_{analysis_id}"],
            "needs_human": result.needs_human,
            "source": result.response_source,
        }
    )
    st.session_state.approved_ids.add(analysis_id)
    st.session_state.toast = ("Draft approved. It was not sent.", "✅")


init_state()
inject_style()
settings = get_settings()
llm = get_llm()

# --- Barre latérale -------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    if llm is not None:
        use_llm = st.checkbox("Use the LLM to write the draft", value=True, key="use_llm")
        st.caption(f"Model: `{settings.model}`")
    else:
        use_llm = False
        st.info("No LLM configured: safe template replies are used. Add your API key to `.env`.")
    st.divider()
    st.metric("Messages in this conversation", len(st.session_state.history))
    st.button("New conversation", on_click=new_conversation, key="new_conversation")
    st.metric("Drafts approved (never sent)", len(st.session_state.approved))
    st.caption("Synthetic data only. Nothing is ever sent to a real customer.")

# --- Saisie ---------------------------------------------------------------------
st.html(masthead_html())

if st.session_state.toast:
    message, icon = st.session_state.toast
    st.session_state.toast = None
    st.toast(message, icon=icon)

st.selectbox(
    "Load an example",
    ["—", *SAMPLE_MESSAGES],
    key="example",
    on_change=load_example,
)
st.text_area(
    "Customer message",
    key="message_input",
    height=120,
    max_chars=MAX_MESSAGE_LENGTH,
    placeholder="Type or paste the customer's message here...",
)

if st.button("Analyze message", type="primary", key="analyze"):
    text = st.session_state.message_input.strip()
    if not text:
        st.warning("Enter a customer message first.")
    else:
        try:
            with st.spinner("Reading the message and checking the rules..."):
                analysis = analyze_message(
                    text,
                    kb=get_kb(),
                    llm=llm if use_llm else None,
                    history=st.session_state.history,
                )
        except ValidationError:
            st.error(f"The message is invalid (empty or longer than {MAX_MESSAGE_LENGTH} characters).")
        except Exception:  # filet de sécurité : l'interface ne doit pas planter
            st.error("Unexpected error while analysing the message. Please try again.")
        else:
            st.session_state.history.append(text)
            st.session_state.result = analysis
            st.session_state.analysis_id += 1
            st.session_state.toast = (
                ("This case needs a human agent.", "🚨")
                if analysis.needs_human
                else ("Draft ready for your review.", "📝")
            )
            st.rerun()

if st.session_state.history:
    with st.expander(f"Conversation so far ({len(st.session_state.history)} message(s))"):
        for number, old in enumerate(st.session_state.history, start=1):
            st.text(f"{number}. {old}")  # st.text : pas d'interprétation Markdown/HTML

# --- Résultat -------------------------------------------------------------------
result = st.session_state.result
if result is None:
    st.html(empty_state_html())
    st.stop()

st.html(ticket_html(result, st.session_state.analysis_id))

tab_reply, tab_kb, tab_case = st.tabs(
    ["Suggested response", "Knowledge base", "Case summary"]
)

with tab_reply:
    source = "written by the LLM" if result.response_source == "llm" else "safe template"
    st.caption(f"Draft source: {source}. Language detected: {result.language}.")
    if result.needs_human:
        st.warning(
            "This case needs a human. The draft is only a holding message: "
            "the AI cannot promise a refund, a cancellation or any other action."
        )
    draft_key = f"draft_{st.session_state.analysis_id}"
    st.text_area(
        "Draft reply (you can edit it)",
        value=result.suggested_response,
        key=draft_key,
        height=220,
    )
    for warning in draft_warnings(st.session_state[draft_key]):
        st.warning(warning)

    already = st.session_state.analysis_id in st.session_state.approved_ids
    st.button(
        "Approve draft (does not send)",
        type="primary",
        on_click=approve_draft,
        disabled=already,
        key="approve",
    )
    if already:
        st.success("Draft approved and saved in this session. It has NOT been sent to anyone.")

with tab_kb:
    if not result.retrieved_information:
        st.info("No reliable information found in the knowledge base. Nothing was invented.")
    for info in result.retrieved_information:
        with st.expander(f"{info.doc_id} - {info.title}  (relevance {info.score:.0%})"):
            st.html(relevance_bar_html(info.score))
            st.text(info.content)

with tab_case:
    if result.handoff is None:
        st.info("No handoff needed: this case is not escalated.")
    else:
        st.code(format_handoff(result.handoff), language=None)

if st.session_state.approved:
    with st.expander(f"Approved drafts ({len(st.session_state.approved)}) - not sent"):
        for number, item in enumerate(st.session_state.approved, start=1):
            st.markdown(f"**{number}.** Customer message:")
            st.text(item["customer_message"])
            st.markdown("Approved reply:")
            st.text(item["reply"])
            st.divider()
