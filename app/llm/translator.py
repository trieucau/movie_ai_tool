import os
import json
import logging
from typing import Optional

from app.utils import get_logger
from app.config import config

logger = get_logger(__name__)

def _call_openai_translation(text: str, target_lang: str = "Vietnamese") -> str:
    """Translate *text* to *target_lang* using OpenAI chat completion.
    Returns the translated string. If the OpenAI API key is missing or the request fails,
    the function falls back to returning the original text.
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed; skipping translation.")
        return text

    if not config.api.openai_api_key:
        logger.warning("OPENAI_API_KEY not set; translation disabled.")
        return text

    client = OpenAI(api_key=config.api.openai_api_key, base_url=config.api.openai_base_url or None)
    system_prompt = "You are a professional translator. Translate the given English text into clear, natural Vietnamese, preserving the meaning and style. Return only the translated text without any explanation."
    try:
        response = client.chat.completions.create(
            model=config.api.openai_model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
            temperature=0.0,
            max_tokens=2000,
        )
        translated = response.choices[0].message.content.strip()
        return translated
    except Exception as e:
        logger.error(f"OpenAI translation failed: {e}")
        return text


def translate_to_vietnamese(text: str) -> str:
    """Public helper used by the pipeline.
    Translates *text* (typically the transcript) to Vietnamese. The function logs
    the timing and returns the translated string. If translation cannot be performed,
    the original *text* is returned unchanged.
    """
    logger.info("Translating transcript to Vietnamese...")
    translated = _call_openai_translation(text, target_lang="Vietnamese")
    if translated != text:
        logger.info("Translation completed successfully.")
    else:
        logger.info("Translation skipped or failed; using original text.")
    return translated
