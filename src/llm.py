from google import genai
from google.genai import types

import config


class LLMAgent:
    def __init__(self):
        if not config.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY tapılmadı. .env faylını yaradıb açarı yaz."
            )

        self.client = genai.Client(api_key=config.GEMINI_API_KEY)

        self.chat = self.client.chats.create(
            model=config.LLM_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=config.SYSTEM_PROMPT,
                temperature=config.LLM_TEMPERATURE,
                max_output_tokens=config.LLM_MAX_OUTPUT_TOKENS,
            ),
        )

    def reply(self, user_text: str) -> str:
        response = self.chat.send_message(user_text)
        return (response.text or "").strip()