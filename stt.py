"""
stt.py — Speech-to-text transcription using Google Web Speech API (no API key required).

Uses the SpeechRecognition library which calls Google's free Web Speech API.
Supports Hindi, Marathi, and English audio.
"""

import os

import speech_recognition as sr


def transcribe_audio(audio_file_path: str) -> str:
    """
    Transcribe a WAV audio file to text.

    Tries Hindi first, then English as fallback. Gradio saves mic recordings
    as WAV files and passes the path directly here.

    Returns the transcribed string, or an empty string on failure.
    """
    if not audio_file_path:
        return ""

    if not os.path.exists(audio_file_path):
        print(f"[stt.py] Audio file not found: {audio_file_path}")
        return ""

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(audio_file_path) as source:
            # Adjust for ambient noise briefly, then record the full clip
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio_data = recognizer.record(source)
    except Exception as e:
        print(f"[stt.py] Failed to read audio file: {e}")
        return ""

    # Try Hindi/Marathi first (Devanagari script speakers), then English
    for lang_code in ("hi-IN", "en-IN"):
        try:
            text = recognizer.recognize_google(audio_data, language=lang_code)
            print(f"[stt.py] Transcribed ({lang_code}): {text}")
            return text.strip()
        except sr.UnknownValueError:
            print(f"[stt.py] Could not understand audio in {lang_code}, trying next.")
        except sr.RequestError as e:
            print(f"[stt.py] Google Speech API request failed: {e}")
            break
        except Exception as e:
            print(f"[stt.py] Unexpected error: {e}")
            break

    print("[stt.py] All transcription attempts failed.")
    return ""
