import streamlit as st
from streamlit_chat import message
import os
import time
from audio_recorder_streamlit import audio_recorder
from streamlit_float import float_init
import base64

# --------------------------------------------------------------------------------------------------------------------------logic2END
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
import base64
import streamlit as st
import openai
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
import chardet
import sys
import json
import re
from pathlib import Path
import importlib
import screening as screening_module
importlib.reload(screening_module)
from screening import (
    PHASE_CONVERSATION,
)

st.set_page_config(page_title="HopeBot: Your Mental Health Assistant", layout="wide")
try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3
sys.modules["sqlite3"] = sqlite3
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent

try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error(
        "OPENAI_API_KEY is missing. Add it to .streamlit/secrets.toml or to a .env file, then restart Streamlit."
    )
    st.stop()

openai.api_key = openai_api_key
openai_chat_model = st.secrets.get("OPENAI_CHAT_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o"))


def detect_file_encoding(file_path: Path) -> str:
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read())
    return result.get("encoding") or "utf-8"


def load_or_create_vectorstore(source_file: str, persist_dir_name: str, collection_name: str, embed_model):
    persist_dir = BASE_DIR / persist_dir_name
    source_path = BASE_DIR / source_file

    if (persist_dir / "chroma.sqlite3").exists():
        return Chroma(
            embedding_function=embed_model,
            persist_directory=str(persist_dir),
            collection_name=collection_name,
        )

    encoding = detect_file_encoding(source_path)
    loader = TextLoader(str(source_path), encoding=encoding)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    return Chroma.from_documents(
        documents=splits,
        embedding=embed_model,
        collection_name=collection_name,
        persist_directory=str(persist_dir),
    )


# Function to initialize resources
@st.cache_resource
def initialize_resources():
    # Chat model
    chat = ChatOpenAI(
        model=openai_chat_model,
        temperature=0.2,
    )

    # Embedding model
    embed_model = OpenAIEmbeddings()

    # Vector stores are rebuilt from source text on Streamlit Cloud if the
    # persisted Chroma directories are not present.
    vectorstore1 = load_or_create_vectorstore(
        source_file="cleaned_data.txt",
        persist_dir_name="cleaned_data",
        collection_name="cleaned_data_docs",
        embed_model=embed_model,
    )
    vectorstore2 = load_or_create_vectorstore(
        source_file="mental_health_support.txt",
        persist_dir_name="mental_health",
        collection_name="mental_health_docs",
        embed_model=embed_model,
    )
    vectorstore3 = load_or_create_vectorstore(
        source_file="econ_example.txt",
        persist_dir_name="econ",
        collection_name="econ_docs",
        embed_model=embed_model,
    )

    # Retrievers
    retriever1 = vectorstore1.as_retriever(k=2)
    retriever2 = vectorstore2.as_retriever(k=2)
    retriever3 = vectorstore3.as_retriever(k=2)

    # ChatPromptTemplate
    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
You are HopeBot, a patient, respectful, and professional mental health counsellor. Your responses are spoken aloud, so keep them warm, natural, concise, and conversational. Do not sound like a menu or a triage bot.

You must guide the conversation through three stages while preserving natural flow:

Stage 1 - supportive conversation:
- Begin with warmth and listen like a counsellor. Reflect the user's feelings and ask one gentle follow-up question at a time.
- Keep this stage within 15 user turns maximum. Do not artificially stretch the conversation.
- If the user says they do not want to talk, have nothing to share, are done, or gives very minimal replies, give one low-pressure invitation to share a little more. If they still do not want to talk, naturally ask whether they would be willing to do a brief screening questionnaire.
- If the user clearly shows depression, anxiety, or manic/hypomanic tendency, remember that tendency, but still finish the first-stage conversation naturally before recommending the relevant scale.
- If symptoms are mixed or unclear, do not force a recommendation. Briefly explain the three options and ask which feels most relevant.

Stage 2 - scale selection and questionnaire:
- The available scales are PHQ-9 for depression symptoms, GAD-7 for anxiety symptoms, and MDQ for manic or hypomanic symptoms.
- If depression is the clearest signal, recommend PHQ-9. If anxiety is clearest, recommend GAD-7. If manic/hypomanic symptoms are clearest, recommend MDQ.
- If the user asks whether they must do it, explain that it is optional and can help clarify support. Ask for consent naturally.
- Once the user chooses or agrees to a scale, ask the scale questions one at a time and in order.
- For PHQ-9 and GAD-7, classify answers as 0 = not at all, 1 = several days, 2 = more than half the days, 3 = nearly every day.
- For MDQ, classify each symptom item as 0 = no or 1 = yes.
- If the user says they do not know, are unsure, or do not understand a question, explain the wording and help them choose the closest answer. If you summarize their meaning, ask them to confirm before scoring.
- Do not reveal the running score while the questionnaire is ongoing.

Stage 3 - results:
- After all items are answered, summarize how each answer was interpreted, give the total score, explain the severity/range, and give brief supportive next steps.
- Be clear that you are a virtual mental health assistant, not a doctor, and this is not a diagnosis or a substitute for professional care.
- If risk or severe symptoms appear, encourage professional support and urgent help if they cannot stay safe.

Hidden JSON for app scoring:
When and only when you have just classified a questionnaire item response, append one final separate line exactly like this:
###JSON_START###{{"scale_id":"phq9|gad7|mdq","question_index":1,"answer_label":"not at all|several days|more than half the days|nearly every day|yes|no","score":0}}###JSON_END###
Use 1-based question_index. For PHQ-9 and GAD-7 score must be 0, 1, 2, or 3. For MDQ score must be 0 or 1. Never output this JSON outside questionnaire item classification turns.

Use this runtime context and retrieved background material to guide the response:

{context}
            """),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    # Create the LLM chain with the language model and the prompt
    document_chain = LLMChain(llm=chat, prompt=question_answering_prompt)

    # Return all initialized resources
    return chat, retriever1, retriever2, retriever3, question_answering_prompt, document_chain

# Initialize resources (runs once and caches results)
chat, retriever1, retriever2, retriever3, question_answering_prompt, document_chain = initialize_resources()
JSON_START = "###JSON_START###"
JSON_END = "###JSON_END###"
def extract_screening_score(text: str):
    """
    Returns (clean_text_without_json, score_data)
    If no JSON found, returns (text, None)
    """
    if JSON_START in text and JSON_END in text:
        try:
            # split only on the last JSON block to be robust
            pre, json_and_after = text.rsplit(JSON_START, 1)
            json_body, post = json_and_after.split(JSON_END, 1)
            clean_text = (pre + post).strip()
            data = json.loads(json_body.strip())
            return clean_text, data
        except Exception:
            # if parsing fails, just hide any leaked JSON line
            cleaned = re.sub(rf"{re.escape(JSON_START)}.*?{re.escape(JSON_END)}", "", text, flags=re.DOTALL).strip()
            return cleaned, None
    return text, None


def record_screening_score(score_data):
    scale_id = score_data.get("scale_id") or st.session_state.get("selected_scale_id") or "phq9"
    if scale_id not in {"phq9", "gad7", "mdq"}:
        return
    try:
        score = int(score_data.get("score"))
    except Exception:
        return
    if scale_id in {"phq9", "gad7"} and score not in {0, 1, 2, 3}:
        return
    if scale_id == "mdq" and score not in {0, 1}:
        return

    try:
        question_index = int(score_data.get("question_index", len(st.session_state.screening_answers.get(scale_id, [])) + 1))
    except Exception:
        question_index = len(st.session_state.screening_answers.get(scale_id, [])) + 1

    st.session_state.selected_scale_id = scale_id
    st.session_state.screening_answers.setdefault(scale_id, [])
    if any(item.get("question_index") == question_index for item in st.session_state.screening_answers[scale_id]):
        return
    st.session_state.screening_answers[scale_id].append(
        {
            "question_index": question_index,
            "answer_label": score_data.get("answer_label"),
            "score": score,
        }
    )
    st.session_state.screening_scores[scale_id] = st.session_state.screening_scores.get(scale_id, 0) + score


def build_chat_history(messages):
    chat_history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            chat_history.add_message(HumanMessage(content=message["content"]))
        else:
            chat_history.add_message(AIMessage(content=message["content"]))
    return chat_history


# Function to process input and return the chatbot's response
def get_assistant_response(messages, stage_context=""):
    # Extract the user's last message (the latest user input)
    user_input = messages[-1]["content"]

    # Simulate chat history
    chat_history = build_chat_history(messages)

    # Retrieve documents based on user input
    retriever_context = user_input  # Use user input as the query for document retrieval
    retrieved_docs1 = retriever1.get_relevant_documents(retriever_context)
    retrieved_docs2 = retriever2.get_relevant_documents(retriever_context)
    retrieved_docs3 = retriever3.get_relevant_documents(retriever_context)

    # Combine retrieved content into one context
    retrieved_context = "\n".join([doc.page_content for doc in retrieved_docs1 + retrieved_docs2 + retrieved_docs3])
    combined_context = f"{stage_context}\n\nRetrieved support material:\n{retrieved_context}".strip()

    # Generate chatbot response with retrieved context
    response = document_chain.run(
        {
            "context": combined_context,  # Documents retrieved from retrievers
            "messages": chat_history.messages  # Conversation history
        }
    )

    # Return the assistant's response
    return response


def speech_to_text(audio_data):
    with open(audio_data, "rb") as audio_file:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file
        )
    return transcript

def text_to_speech(input_text):
    response = openai.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=input_text
    )
    webm_file_path = "temp_audio_play.mp3"
    with open(webm_file_path, "wb") as f:
        response.stream_to_file(webm_file_path)
    return webm_file_path

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)
# ------------------------------------------------------------------------------------------------------------------------------------------------logic2END

# Float feature initialization


def render_typewriter(text, delay=0.018):
    container = st.empty()
    rendered = ""
    for char in text:
        rendered += char
        container.markdown(
            f"<p style='font-size: 24px; margin: 0;'>{rendered}</p>",
            unsafe_allow_html=True,
        )
        time.sleep(delay)

def submit_typed_message():
    text = st.session_state.get("typed_message_input", "").strip()
    if text:
        st.session_state.pending_text_message = text
        st.session_state.typed_message_input = ""

float_init()

# 初始化会话状态
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "This is HopeBot, your mental health assistant. It's good to hear from you, how are you doing today? 😊"}
        ]
    if "total_phq9_score" not in st.session_state:
        st.session_state.total_phq9_score = 0
    if "answers_record" not in st.session_state:
        st.session_state.answers_record = []  # e.g., ["A","B",...]
    if "screening_scores" not in st.session_state:
        st.session_state.screening_scores = {"phq9": 0, "gad7": 0, "mdq": 0}
    if "screening_answers" not in st.session_state:
        st.session_state.screening_answers = {"phq9": [], "gad7": [], "mdq": []}
    if "phase" not in st.session_state:
        st.session_state.phase = PHASE_CONVERSATION
    if "conversation_user_turns" not in st.session_state:
        st.session_state.conversation_user_turns = 0
    if "reluctance_count" not in st.session_state:
        st.session_state.reluctance_count = 0
    if "inferred_scale_id" not in st.session_state:
        st.session_state.inferred_scale_id = None
    if "selected_scale_id" not in st.session_state:
        st.session_state.selected_scale_id = None
    if "screening_session" not in st.session_state:
        st.session_state.screening_session = None
    if "prompt_directive" not in st.session_state:
        st.session_state.prompt_directive = "continue"
    if "current_scale_question" not in st.session_state:
        st.session_state.current_scale_question = ""

initialize_session_state()

# 标题
st.title("HopeBot: Your Mental Health Assistant 🤖")

st.markdown(
    """
    <style>
    .stTextInput input {
        min-height: 64px;
        font-size: 22px;
        border-radius: 14px;
        padding: 0 18px;
    }
    .stTextInput input::placeholder {
        font-size: 18px;
    }
    audio {
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# 语音识别功能
def speech_to_text(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1", response_format="text", file=audio_file
        )
    return transcript.strip()

# 语音合成功能
def text_to_speech(text):
    response = openai.audio.speech.create(model="tts-1", voice="nova", input=text)
    audio_path = "response_audio.mp3"
    with open(audio_path, "wb") as f:
        response.stream_to_file(audio_path)
    return audio_path

# 音频播放功能
def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    b64_audio = base64.b64encode(data).decode("utf-8")
    st.markdown(
        f"""
        <audio controls autoplay id="hopebot_audio">
        <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
        </audio>
        """,
        unsafe_allow_html=True,
    )

# 浮动容器（用于麦克风）
float_init()
footer_container = st.container()
with footer_container:
    mic_col, text_col = st.columns([1, 8], vertical_alignment="center")
    with mic_col:
        audio_bytes = audio_recorder(
            text="",
            icon_size="2x",
            energy_threshold=(-1, 0.5),
            pause_threshold=30,
            sample_rate=30000,
        )
    with text_col:
        st.text_input(
            "Type your message here",
            key="typed_message_input",
            label_visibility="collapsed",
            placeholder="Type your message here, or use the microphone.",
            on_change=submit_typed_message,
        )

# 显示聊天历史（使用气泡样式和头像）
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "🤗"):
        st.markdown(
            f"<p style='font-size: 24px; margin: 0;'>{message['content']}</p>",
            unsafe_allow_html=True
        )

# 处理语音输入
user_message = None

if st.session_state.get("pending_text_message"):
    user_message = st.session_state.pending_text_message
    st.session_state.pending_text_message = None

if audio_bytes and not user_message:
    with st.spinner("Transcribing..."):
        audio_path = "temp_audio.mp3"
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        transcript = speech_to_text(audio_path)
        if transcript:
            user_message = transcript
        os.remove(audio_path)

if user_message:
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(
            f"<p style='font-size: 24px; margin: 0;'>{user_message}</p>",
            unsafe_allow_html=True
        )

# 生成 HopeBot 回复
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking 🤔..."):
            final_response = get_assistant_response(st.session_state.messages)

        cleaned_text, score_data = extract_screening_score(final_response)

        # 默认展示“清理掉 JSON 的文本”；若未检测到 JSON，就展示原文
        display_text = cleaned_text if cleaned_text is not None else final_response

        # 如果有分类与得分，记录并累加
        if score_data is not None:
            record_screening_score(score_data)

        with st.spinner("HopeBot is speaking 💬..."):
            audio_file = text_to_speech(display_text)

        # 同时显示文本和播放音频
        render_typewriter(display_text)
        autoplay_audio(audio_file)  # 播放音频 

        # 添加回复到会话状态
        st.session_state.messages.append({"role": "assistant", "content": display_text})
        os.remove(audio_file)

# 浮动的麦克风按钮
st.markdown("<div style='height: 130px;'></div>", unsafe_allow_html=True)
footer_container.float("bottom: 1.5rem;")
