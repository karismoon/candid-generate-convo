# streamlit_app.py
"""
Streamlit app that lets a human:
1. Choose an LLM provider/model
2. Describe a persona to emulate
3. Have that LLM act as a human client talking to the Intake API
4. Run the full intake flow to completion
5. Save the transcript (role, content) as CSV

Assumptions:
- Intake API is running at http://localhost:9001
- Journey + intake session already exist (journey_id provided)
"""

import csv
import io
import time
from typing import List, Dict

import requests
import streamlit as st

# -----------------------------
# LLM CLIENTS (simple wrappers)
# -----------------------------

def call_openai(model: str, system_prompt: str, messages: List[Dict[str, str]]) -> str:
    from openai import OpenAI

    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}] + messages,
    )
    return completion.choices[0].message.content


def call_anthropic(model: str, system_prompt: str, messages: List[Dict[str, str]]) -> str:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        system=system_prompt,
        messages=messages,
        max_tokens=1024,
    )
    return response.content[0].text


# -----------------------------
# STREAMLIT UI
# -----------------------------

st.title("🤖 Agent-to-Agent Intake Tester")

api_base = st.text_input("Intake API base URL", value="http://localhost:9001")
journey_id = st.text_input("Journey ID (must already exist)")

st.subheader("Client Agent Configuration")

provider = st.selectbox("LLM Provider", ["OpenAI", "Anthropic"])

if provider == "OpenAI":
    model = st.selectbox("Model", ["gpt-4o", "gpt-4.1"])
else:
    model = st.selectbox("Model", ["claude-3-5-sonnet-20240620"])

persona = st.text_area(
    "Describe the human persona to emulate",
    placeholder="e.g. A stressed graduate student after a difficult meeting with their advisor",
)

start_message = st.text_input(
    "Initial message to start intake",
    value="I'm feeling overwhelmed and not sure how to process everything that happened.",
)

run_button = st.button("▶ Run Full Intake Conversation")


# -----------------------------
# CORE LOGIC
# -----------------------------

if run_button:
    if not journey_id or not persona:
        st.error("Journey ID and persona description are required")
        st.stop()

    transcript: List[Dict[str, str]] = []

    system_prompt = f"""
You are role-playing a HUMAN going through an emotional intake conversation.

Persona:
{persona}

Guidelines:
- Speak naturally, like a real person
- Answer questions honestly but imperfectly
- Do NOT act like a therapist or analyst
- Respond with 2–5 sentences max
- Stay emotionally consistent with the persona
"""

    def llm_reply(chat_history):
        if provider == "OpenAI":
            return call_openai(model, system_prompt, chat_history)
        else:
            return call_anthropic(model, system_prompt, chat_history)

    # ---- Start intake ----
    start_resp = requests.post(
        f"{api_base}/api/intake-sessions/start",
        json={
            "journey_id": journey_id,
            "initial_user_message": start_message,
        },
        timeout=60,
    )

    start_resp.raise_for_status()
    data = start_resp.json()

    session_id = data["session_id"]
    messages = data["messages"]

    for m in messages:
        transcript.append({"role": m["role"], "content": m["content"]})

    chat_history = [{"role": "assistant", "content": messages[-1]["content"]}]

    current_step = "intake"

    # ---- Conversation loop ----
    while current_step != "completed":
        user_text = llm_reply(chat_history)

        transcript.append({"role": "user", "content": user_text})

        resp = requests.post(
            f"{api_base}/api/intake-sessions/{session_id}/message",
            json={"message": user_text},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        assistant_text = data["content"]
        current_step = data.get("current_step", "unknown")

        transcript.append({"role": "assistant", "content": assistant_text})

        chat_history = [{"role": "assistant", "content": assistant_text}]

        time.sleep(0.5)  # be polite to the API

    st.success("Intake completed!")

    # -----------------------------
    # CSV EXPORT
    # -----------------------------

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["role", "content"])
    writer.writeheader()
    writer.writerows(transcript)

    st.download_button(
        label="⬇ Download conversation CSV",
        data=output.getvalue(),
        file_name="intake_conversation.csv",
        mime="text/csv",
    )

    st.subheader("Transcript Preview")
    st.dataframe(transcript)
