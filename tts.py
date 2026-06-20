"""
tts.py — Text-to-speech conversion using gTTS (Google Text-to-Speech).
"""

import re

from gtts import gTTS


def detect_language_code(text: str) -> str:
    """
    Detect whether the text is Devanagari-script (Hindi/Marathi) or English.

    Returns "hi" if Devanagari Unicode characters are found, otherwise "en".
    This is used to auto-select the correct TTS language.
    """
    # Devanagari Unicode block: U+0900 to U+097F
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    return "en"


def speak(text: str, lang: str = "hi", output_path: str = "output_audio.mp3") -> str:
    """
    Convert text to speech and save the audio file.

    Args:
        text:        The text to convert to speech.
        lang:        Language code — "hi" (Hindi), "mr" (Marathi), "en" (English).
                     Defaults to "hi" (Hindi).
        output_path: File path where the MP3 will be saved.

    Returns:
        The output_path where the audio was saved, or an empty string on failure.
    """
    if not text or not text.strip():
        return ""

    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(output_path)
        print(f"[tts.py] Audio saved to '{output_path}' (lang={lang})")
        return output_path
    except Exception as e:
        print(f"[tts.py] TTS generation failed: {e}")
        # gTTS requires internet — return empty string so the caller can degrade gracefully
        return ""
