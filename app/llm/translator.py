"""
translator.py — Robust batch translator.

Strategy:
  1. Send segments in small chunks (20 per request) to stay well under token limits.
  2. Each segment gets a stable string key so the model can return them in any order.
  3. Validate every returned segment: if a segment is missing OR still appears to be
     English (heuristic check), retry that specific segment individually.
  4. All retries use exponential back-off.
  5. The function ALWAYS returns the same number of segments as the input; segments
     that could not be translated fall back to the original text rather than being lost.
"""

import json
import time
import copy
import re
from typing import List, Optional, Callable

from app.utils import get_logger
from app.config import config

logger = get_logger(__name__)

# ────────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT (tightened to stop the model from answering in English)
# ────────────────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "Bạn là chuyên gia dịch thuật chuyên nghiệp Việt Nam. "
    "Nhiệm vụ: Dịch TOÀN BỘ văn bản tiếng Anh sang tiếng Việt tự nhiên, chuẩn xác. "
    "Quy tắc bắt buộc:\n"
    "1. Dịch 100% sang tiếng Việt — KHÔNG giữ lại từ tiếng Anh nào (ngoại trừ danh từ riêng, "
    "địa danh, thuật ngữ chuyên ngành không có từ Việt tương đương).\n"
    "2. Trả về ĐÚNG cấu trúc JSON gốc, chỉ thay nội dung giá trị — không thêm, không bỏ bất kỳ key nào.\n"
    "3. KHÔNG viết giải thích, KHÔNG kèm markdown, KHÔNG thêm text ngoài JSON.\n"
    "4. Nếu đoạn văn bản đã là tiếng Việt, giữ nguyên."
)

_SINGLE_SYSTEM_PROMPT = (
    "Bạn là chuyên gia dịch thuật chuyên nghiệp Việt Nam. "
    "Dịch đoạn văn bản sau từ tiếng Anh sang tiếng Việt tự nhiên, chuẩn xác. "
    "KHÔNG giữ lại từ tiếng Anh nào (ngoại trừ danh từ riêng, địa danh, thuật ngữ chuyên ngành). "
    "Chỉ trả về bản dịch tiếng Việt, KHÔNG giải thích thêm gì."
)

# ────────────────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────────────────

def _build_client():
    """Create an OpenAI client, or return None if not configured."""
    try:
        from openai import OpenAI
    except ImportError:
        return None
    if not config.api.openai_api_key:
        return None
    return OpenAI(
        api_key=config.api.openai_api_key,
        base_url=config.api.openai_base_url or None,
    )


def _is_vietnamese(text: str) -> bool:
    """
    Returns True if the text appears to already be Vietnamese.
    Vietnamese has a rich set of unique diacritics not found in other languages.
    A segment with >=1 Vietnamese diacritic per 25 characters is considered Vietnamese.
    Very short segments (< 6 chars) without diacritics are sent for translation anyway.
    """
    stripped = text.strip() if text else ""
    if not stripped:
        return True  # empty — skip
    
    vi_pattern = re.compile(
        r'[àáâãèéêìíòóôõùúýăđơư'
        r'ạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ'
        r'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯ'
        r'ẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ]'
    )
    vi_chars = len(vi_pattern.findall(stripped))
    
    if vi_chars == 0:
        # No Vietnamese diacritics at all — definitely not Vietnamese
        # (unless extremely short like "OK", numbers, punctuation only)
        letter_count = sum(1 for c in stripped if c.isalpha())
        if letter_count <= 2:
            return True  # too short/ambiguous (e.g. "OK", "hi"), don't bother re-translating
        return False
    
    # Has Vietnamese diacritics — check density
    return vi_chars >= max(1, len(stripped) / 25)



def _looks_untranslated(text: str) -> bool:
    """Returns True if the text does NOT appear to be Vietnamese (needs translation)."""
    return not _is_vietnamese(text)


def _translate_single(client, text: str, segment_idx: int) -> str:
    """Translate a single text string with up to 3 retries. Returns original on failure."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=config.api.openai_model,
                messages=[
                    {"role": "system", "content": _SINGLE_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            result = response.choices[0].message.content.strip()
            if result:
                logger.debug(f"Single-translate segment {segment_idx}: OK")
                return result
        except Exception as e:
            wait = 2 ** (attempt + 1)
            logger.warning(f"Single-translate segment {segment_idx} attempt {attempt+1} failed: {e}. Retry in {wait}s")
            time.sleep(wait)
    logger.error(f"Single-translate segment {segment_idx} permanently failed — keeping original.")
    return text


def _translate_chunk(client, payload: dict, chunk_start_idx: int) -> dict:
    """
    Translate a dict {str(global_idx): text} to Vietnamese.
    Returns a dict with the same keys, values replaced by Vietnamese.
    Raises on unrecoverable failure.
    """
    user_prompt = f"Dịch các đoạn sau sang tiếng Việt:\n{json.dumps(payload, ensure_ascii=False)}"
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=config.api.openai_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            data = json.loads(raw)
            if isinstance(data, dict) and len(data) > 0:
                return data
            raise ValueError(f"Empty or non-dict response: {raw[:100]}")
        except Exception as e:
            wait = 2 ** (attempt + 1)
            logger.warning(f"Chunk translate (start={chunk_start_idx}) attempt {attempt+1} failed: {e}. Retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Chunk translate (start={chunk_start_idx}) permanently failed.")


# ────────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ────────────────────────────────────────────────────────────────────────────────

def batch_translate_segments(
    segments: list,
    progress_callback: Optional[Callable] = None,
) -> list:
    """
    Translate all segments to Vietnamese with guaranteed coverage.
    
    Algorithm:
      Phase 1 — Batch: send 20 segments at a time. Collect translated data.
      Phase 2 — Validation: for every segment that is still English (heuristic),
                re-translate individually.
    
    Returns a deep-copy of `segments` with .text replaced by Vietnamese.
    """
    client = _build_client()
    if client is None:
        logger.warning("OpenAI not configured — translation skipped.")
        return segments

    translated_segments = copy.deepcopy(segments)
    total = len(segments)
    chunk_size = 20  # small enough to be safe, large enough to be fast
    total_chunks = (total + chunk_size - 1) // chunk_size

    logger.info(f"Starting batch translation: {total} segments → {total_chunks} chunks of {chunk_size}")
    logger.info("Rule: ALL non-Vietnamese segments will be translated. Vietnamese segments will be kept as-is.")

    # ── Phase 1: Batch translation ──────────────────────────────────────────────
    for chunk_num, i in enumerate(range(0, total, chunk_size)):
        chunk = segments[i : i + chunk_size]
        if progress_callback:
            progress_callback(
                0.40 + 0.18 * (i / total),
                f"Dịch batch {chunk_num + 1}/{total_chunks}...",
            )

        # Only include segments that are NOT already Vietnamese
        payload = {
            str(i + j): seg.text
            for j, seg in enumerate(chunk)
            if seg.text.strip() and not _is_vietnamese(seg.text)
        }

        if not payload:
            continue

        try:
            translated_data = _translate_chunk(client, payload, i)
            for key, val in translated_data.items():
                try:
                    idx = int(key)
                    if 0 <= idx < total and val and val.strip():
                        translated_segments[idx].text = val.strip()
                        logger.debug(f"  Segment {idx} translated OK")
                except (ValueError, KeyError):
                    pass
        except RuntimeError as e:
            logger.error(f"Chunk {chunk_num+1} failed entirely: {e}")
            logger.warning(f"Will retry chunk {chunk_num+1} segment-by-segment in Phase 2.")

    # ── Phase 2: Validate & fix remaining untranslated segments ─────────────────
    if progress_callback:
        progress_callback(0.58, "Kiểm tra và dịch lại đoạn còn sót...")

    need_retry = [
        idx for idx, seg in enumerate(translated_segments)
        if seg.text.strip() and not _is_vietnamese(seg.text)
    ]

    if need_retry:
        logger.info(f"Phase 2: {len(need_retry)} segments still NOT Vietnamese — retrying individually.")
        for pos, idx in enumerate(need_retry):
            original_text = segments[idx].text
            translated_segments[idx].text = _translate_single(client, original_text, idx)
            if progress_callback:
                progress_callback(
                    0.58 + 0.02 * (pos / max(len(need_retry), 1)),
                    f"Dịch lại đoạn {pos+1}/{len(need_retry)}...",
                )
    else:
        logger.info("Phase 2: All segments look translated. ✓")

    # ── Final report ─────────────────────────────────────────────────────────────
    still_not_vi = sum(1 for s in translated_segments if s.text.strip() and not _is_vietnamese(s.text))
    logger.info(f"[TRANSLATE] Complete: {total - still_not_vi}/{total} segments now in Vietnamese.")
    if still_not_vi:
        logger.warning(f"[TRANSLATE] {still_not_vi} segment(s) could not be translated — kept original text.")

    return translated_segments


def translate_to_vietnamese(text: str) -> str:
    """Translate a single block of text to Vietnamese (used for non-segment workflows)."""
    client = _build_client()
    if client is None:
        logger.warning("OpenAI not configured — translation skipped.")
        return text
    logger.info("Translating text block to Vietnamese...")
    result = _translate_single(client, text, -1)
    if result != text:
        logger.info("Text block translation completed successfully.")
    else:
        logger.warning("Text block translation returned original text.")
    return result
