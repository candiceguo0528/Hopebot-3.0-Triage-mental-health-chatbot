# HopeBot-Candice

HopeBot is a Streamlit mental health support chatbot with RAG, text input, voice input, OpenAI speech-to-text, OpenAI text-to-speech, and PHQ-9/GAD-7/MDQ screening flow support.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app0.py
```

Create `.streamlit/secrets.toml` locally:

```toml
OPENAI_API_KEY = "your-openai-api-key"
# Optional. Defaults to gpt-5.5 if omitted.
OPENAI_CHAT_MODEL = "gpt-5.5"
```

Do not commit `.streamlit/secrets.toml`.

## Deploy On Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app from this repository.
3. Set the main file path to `app0.py`.
4. Add this secret in the app settings:

```toml
OPENAI_API_KEY = "your-openai-api-key"
# Optional. Defaults to gpt-5.5 if omitted.
OPENAI_CHAT_MODEL = "gpt-5.5"
```

The Chroma vector stores are rebuilt from the source text files on first startup if the persisted local database folders are not present. Chat responses use the OpenAI Responses API so GPT-5.5 can be used directly.

## Notes

This project is for supportive screening and educational interaction. It is not a diagnosis tool and is not a substitute for professional mental health care.
