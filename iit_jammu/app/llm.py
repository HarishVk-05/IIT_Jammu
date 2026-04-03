from __future__ import annotations
import logging

from langchain_groq import ChatGroq
from iit_jammu.config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

import streamlit as st

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") 
def get_llm(temperature: float = 0.0) -> ChatGroq:
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is missing. Add it to your .env file and restart."
            )
    return ChatGroq(
        model = GROQ_MODEL,
        temperature = temperature,
        max_retries = 3,
        timeout = 60,
        api_key = GROQ_API_KEY
    )
