"""
app.py — Farming Consultant with RAG-based multi-turn chatbot.

RAG strategy:
  Retrieve  — last N turns matching the current intent
  Augment   — inject retrieved context + image analysis into prompt
  Generate  — LLM produces a grounded, context-aware response
"""

import os

import gradio as gr
from dotenv import load_dotenv

from disease_module import handle_disease_query, lookup_disease_info
from llm import analyze_image_with_vision, call_llm, classify_intent, detect_language
from market_module import handle_market_query
from prompts import SYSTEM_PROMPT_BASE
from stt import transcribe_audio
from tts import speak

load_dotenv()

LANG_CODE_TO_CHOICE = {"hi": "Hindi", "mr": "Marathi", "en": "English"}

OFF_TOPIC_MSG = {
    "hi": (
        "मैं सिर्फ फसल रोग और मंडी भाव के बारे में मदद कर सकता हूँ। "
        "कृपया फसल रोग, कीट, या बिक्री से जुड़ा सवाल पूछें।"
    ),
    "mr": (
        "मी फक्त पिक रोग आणि मंडी भावाबद्दल मदत करू शकतो. "
        "कृपया पिक रोग, किडा, किंवा विक्रीबद्दल प्रश्न विचारा."
    ),
    "en": (
        "I can only help with crop diseases and market information. "
        "Please ask about a plant disease, pest problem, or crop selling price."
    ),
}

RAG_WINDOW = 4


def _retrieve_relevant_context(llm_history: list, window: int = RAG_WINDOW) -> list:
    if not llm_history:
        return []
    total_msgs = len(llm_history)
    window_msgs = window * 2
    if total_msgs <= window_msgs:
        return llm_history
    anchor = llm_history[:2]
    recent = llm_history[-(window_msgs - 2) :]
    if llm_history[0] in recent:
        return recent
    return anchor + recent


def chat_fn(
    audio_input,
    image_input,
    text_input,
    language_choice,
    chatbot_history,
    llm_history,
    cached_image,
):
    query_text = ""
    if audio_input is not None:
        query_text = transcribe_audio(audio_input)
        if not query_text:
            print("[app.py] STT failed — using text fallback.")

    if not query_text and text_input and text_input.strip():
        query_text = text_input.strip()

    if not query_text:
        return chatbot_history, llm_history, cached_image, "", None, language_choice

    print(f"[app.py] Query: {query_text}")
    detected_lang = detect_language(query_text)
    detected_choice = LANG_CODE_TO_CHOICE.get(detected_lang, "Hindi")
    print(f"[app.py] Language: {detected_lang}")

    image_context = ""
    new_cached = cached_image
    if image_input:
        if image_input == cached_image.get("path"):
            image_context = cached_image.get("analysis", "")
            print("[app.py] Reusing cached image analysis.")
        else:
            print(f"[app.py] Analyzing new image: {image_input}")
            image_context = analyze_image_with_vision(
                image_input, description=query_text
            )
            new_cached = {"path": image_input, "analysis": image_context}
            if image_context:
                print(f"[app.py] Image cached. Preview: {image_context[:100]}...")

    user_msg_for_llm = query_text
    if image_context:
        user_msg_for_llm = (
            f"{query_text}\n\n[Farmer's crop photo analysis: {image_context}]"
        )

    intent = classify_intent(query_text)
    is_first_turn = len(llm_history) == 0
    print(f"[app.py] Intent: {intent} | First turn: {is_first_turn}")

    if intent == "MARKET":
        print("[app.py] Routing → MARKET module (fresh)")
        response_text = handle_market_query(query_text)
    elif intent == "DISEASE" and is_first_turn:
        print("[app.py] Routing → DISEASE module (first turn)")
        if image_context:
            response_text = handle_disease_query(user_msg_for_llm, image_path=None)
        else:
            response_text = handle_disease_query(query_text, image_path=image_input)
    else:
        print(f"[app.py] Routing → RAG (anchor + last {RAG_WINDOW} turns + KB lookup)")
        retrieved = _retrieve_relevant_context(llm_history)
        kb_info = lookup_disease_info(query_text, image_context)
        if kb_info:
            augmented_system = (
                SYSTEM_PROMPT_BASE
                + "\n\nThe following is verified farming knowledge from your knowledge base. "
                + "Use it to answer the farmer's question accurately:\n\n"
                + kb_info
            )
            print("[app.py] KB knowledge injected into prompt.")
        else:
            augmented_system = SYSTEM_PROMPT_BASE
        messages = retrieved + [{"role": "user", "content": user_msg_for_llm}]
        response_text = call_llm(augmented_system, messages=messages)

    llm_history = llm_history + [
        {"role": "user", "content": user_msg_for_llm},
        {"role": "assistant", "content": response_text},
    ]
    chatbot_history = chatbot_history + [[query_text, response_text]]

    audio_path = speak(
        response_text, lang=detected_lang, output_path="output_audio.mp3"
    )
    return (
        chatbot_history,
        llm_history,
        new_cached,
        "",
        audio_path if audio_path else None,
        detected_choice,
    )


def clear_chat():
    return [], [], {"path": None, "analysis": ""}, "", None


# ═══════════════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════════════

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700;800&display=swap');

/* ── RESET ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; }
body, .gradio-container {
    font-family: 'Plus Jakarta Sans', ui-sans-serif, sans-serif !important;
    background: #f1f5f9 !important;
}
.gradio-container { max-width: 100% !important; padding: 0 !important; }
footer { display: none !important; }

/* ── HOME TAB: strip Gradio .block chrome & fix gaps ──
   gr.HTML renders inside a .block div that adds padding / border / shadow.
   Target by contents so only the home tab is affected. */
.block:has(.nw-hero),
.block:has(.nw-hero) > * {
    padding: 0 !important; margin: 0 !important; border: none !important;
    border-radius: 0 !important; box-shadow: none !important;
    background: transparent !important; gap: 0 !important;
}

/* ════════════════════════════════════════
   TAB NAVIGATION
════════════════════════════════════════ */
.km-tabs > div.tab-nav {
    background: white !important;
    border-bottom: 2px solid #e2e8f0 !important;
    padding: 0 40px !important;
    display: flex !important;
    gap: 4px !important;
    position: sticky !important;
    top: 0 !important;
    z-index: 50 !important;
    box-shadow: 0 1px 12px rgba(0,0,0,0.06) !important;
    justify-content: center !important;
}
.km-tabs button[role="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 14px 24px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    transition: all 0.22s !important;
    margin-bottom: -2px !important;
    letter-spacing: 0.01em !important;
    border-radius: 8px 8px 0 0 !important;
}
.km-tabs button[role="tab"]:hover { color: #16a34a !important; background: rgba(22,163,74,0.04) !important; }
.km-tabs button[role="tab"].selected {
    color: #16a34a !important;
    border-bottom-color: #16a34a !important;
    background: linear-gradient(180deg, rgba(22,163,74,0.06) 0%, transparent 100%) !important;
    font-weight: 700 !important;
}
.km-tabs > .tabitem { padding: 0 !important; background: transparent !important; }

/* ════════════════════════════════════════
   HOME — HERO
════════════════════════════════════════ */
.hm-hero {
    background: linear-gradient(135deg, #020d06 0%, #0b2818 20%, #14532d 45%, #16a34a 75%, #22c55e 100%);
    background-size: 400% 400%;
    animation: heroGrad 12s ease-in-out infinite;
    padding: 90px 72px;
    display: grid;
    grid-template-columns: 1fr 420px;
    gap: 60px;
    align-items: center;
    overflow: hidden;
    position: relative;
    min-height: 580px;
}
/* Dot mesh */
.hm-hero::before {
    content: '';
    position: absolute; inset: 0; z-index: 0;
    background-image: radial-gradient(rgba(255,255,255,0.055) 1px, transparent 1px);
    background-size: 24px 24px;
    pointer-events: none;
}
/* Glow blobs */
.hm-hero::after {
    content: '';
    position: absolute; top: -80px; right: 300px; z-index: 0;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(74,222,128,0.18) 0%, transparent 60%);
    border-radius: 50%; pointer-events: none;
}
.hm-hero-left { position: relative; z-index: 2; }
.hm-eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    color: #86efac; font-size: 0.75rem; font-weight: 700;
    padding: 6px 16px; border-radius: 100px;
    margin-bottom: 26px; letter-spacing: 0.08em; text-transform: uppercase;
}
.hm-dot {
    width: 8px; height: 8px; background: #22c55e;
    border-radius: 50%; flex-shrink: 0;
    animation: pulseDot 2s ease-in-out infinite;
}
.hm-h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: clamp(2.6rem, 5vw, 3.8rem);
    font-weight: 800; color: white; line-height: 1.1;
    margin: 0 0 22px; letter-spacing: -0.03em;
}
.hm-grad-text {
    background: linear-gradient(100deg, #86efac, #4ade80, #a3e635, #fbbf24, #86efac);
    background-size: 300% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradText 5s linear infinite;
}
.hm-desc {
    color: rgba(255,255,255,0.75); font-size: 1.05rem;
    line-height: 1.75; margin-bottom: 32px; max-width: 500px;
}
.hm-chips { display: flex; flex-wrap: wrap; gap: 10px; }
.hm-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2);
    color: #dcfce7; font-size: 0.78rem; font-weight: 600;
    padding: 7px 15px; border-radius: 100px; backdrop-filter: blur(6px);
    transition: background 0.2s, transform 0.2s;
}
.hm-chip:hover { background: rgba(255,255,255,0.18); transform: translateY(-1px); }

/* Hero chat preview card */
.hm-hero-right { position: relative; z-index: 2; }
.hm-chat-card {
    background: white; border-radius: 22px;
    box-shadow: 0 40px 100px rgba(0,0,0,0.5), 0 8px 32px rgba(0,0,0,0.25);
    overflow: hidden; width: 100%;
    transform: perspective(800px) rotateY(-4deg) rotateX(2deg);
    animation: floatCard 7s ease-in-out infinite;
    transition: transform 0.4s ease;
}
.hm-chat-card:hover {
    transform: perspective(800px) rotateY(-2deg) rotateX(1deg) translateY(-6px);
}
.hm-card-head {
    background: linear-gradient(135deg, #14532d, #16a34a);
    padding: 14px 18px; display: flex; align-items: center; gap: 12px;
    position: relative; overflow: hidden;
}
.hm-card-head::before {
    content: '';
    position: absolute; inset: 0;
    background-image: radial-gradient(rgba(255,255,255,0.06) 1px, transparent 1px);
    background-size: 16px 16px;
}
.hm-card-avatar {
    width: 40px; height: 40px; background: rgba(255,255,255,0.2);
    border-radius: 50%; display: flex; align-items: center; justify-content: center;
    font-size: 1.15rem; flex-shrink: 0; position: relative; z-index: 1;
}
.hm-card-title { color: white; font-weight: 800; font-size: 0.92rem; position: relative; z-index: 1; }
.hm-card-sub { color: #bbf7d0; font-size: 0.7rem; margin-top: 2px; position: relative; z-index: 1; display: flex; align-items: center; gap: 4px; }
.hm-online-dot {
    width: 7px; height: 7px; background: #4ade80;
    border-radius: 50%; display: inline-block;
    animation: pulseDot 2s ease-in-out infinite;
}
.hm-chat-body {
    padding: 16px 16px 10px; background: #f8fafc;
    display: flex; flex-direction: column; gap: 10px;
}
.hm-msg { max-width: 82%; opacity: 0; animation: msgIn 0.5s ease forwards; }
.hm-msg.user { align-self: flex-end; }
.hm-msg.bot  { align-self: flex-start; }
.hm-msg-bubble {
    padding: 10px 14px; border-radius: 16px; font-size: 0.8rem; line-height: 1.55;
}
.hm-msg.user .hm-msg-bubble {
    background: linear-gradient(135deg, #16a34a, #22c55e);
    color: white; border-bottom-right-radius: 4px;
    box-shadow: 0 4px 12px rgba(22,163,74,0.3);
}
.hm-msg.bot .hm-msg-bubble {
    background: white; color: #1e293b;
    border: 1px solid #e2e8f0; border-bottom-left-radius: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.hm-card-input {
    display: flex; align-items: center; gap: 8px;
    padding: 12px 16px; border-top: 1px solid #f1f5f9; background: white;
}
.hm-card-input-bar {
    flex: 1; background: #f8fafc; border: 1.5px solid #e2e8f0;
    border-radius: 10px; padding: 8px 13px; font-size: 0.73rem; color: #94a3b8;
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.hm-card-send {
    width: 32px; height: 32px; flex-shrink: 0;
    background: linear-gradient(135deg, #16a34a, #22c55e);
    border-radius: 9px; display: flex; align-items: center;
    justify-content: center; font-size: 0.85rem; color: white;
    box-shadow: 0 3px 8px rgba(22,163,74,0.4);
}

/* ════════════════════════════════════════
   HOME — TICKER
════════════════════════════════════════ */
.hm-ticker-wrap {
    background: white; border-top: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0; overflow: hidden;
    padding: 13px 0; position: relative;
}
.hm-ticker-wrap::before, .hm-ticker-wrap::after {
    content: ''; position: absolute; top: 0; bottom: 0; width: 80px; z-index: 2;
}
.hm-ticker-wrap::before { left: 0; background: linear-gradient(90deg, white, transparent); }
.hm-ticker-wrap::after  { right: 0; background: linear-gradient(-90deg, white, transparent); }
.hm-ticker-track {
    display: flex; width: max-content;
    animation: ticker 32s linear infinite;
}
.hm-ticker-item {
    color: #64748b; font-size: 0.85rem; font-weight: 500;
    white-space: nowrap; padding: 0 18px;
}
.hm-ticker-item span { color: #16a34a; font-weight: 700; }

/* ════════════════════════════════════════
   HOME — STATS
════════════════════════════════════════ */
.hm-stats { background: white; padding: 64px 72px; }
.hm-stats-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 24px;
    max-width: 960px; margin: 0 auto;
}
.hm-stat-card {
    background: linear-gradient(145deg, #ffffff, #f8fafc);
    border: 1.5px solid #e2e8f0; border-radius: 20px;
    padding: 32px 24px; text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    transition: transform 0.28s, box-shadow 0.28s;
    position: relative; overflow: hidden;
}
.hm-stat-card::before {
    content: '';
    position: absolute; bottom: 0; left: 0; right: 0; height: 3px;
}
.hm-stat-card:nth-child(1)::before { background: linear-gradient(90deg, #0ea5e9, #38bdf8); }
.hm-stat-card:nth-child(2)::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.hm-stat-card:nth-child(3)::before { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }
.hm-stat-card:nth-child(4)::before { background: linear-gradient(90deg, #16a34a, #22c55e); }
.hm-stat-card:hover {
    transform: translateY(-6px); box-shadow: 0 16px 40px rgba(0,0,0,0.1);
}
.hm-stat-icon {
    width: 56px; height: 56px; border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; margin: 0 auto 16px;
}
.hm-stat-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem; font-weight: 800; color: #0f172a; line-height: 1; margin-bottom: 8px;
}
.hm-stat-label { color: #64748b; font-size: 0.82rem; font-weight: 500; }

/* ════════════════════════════════════════
   HOME — FEATURES
════════════════════════════════════════ */
.hm-features { padding: 80px 72px; background: #f8fafc; }
.hm-section-label {
    font-size: 0.75rem; font-weight: 800; letter-spacing: 0.12em;
    text-transform: uppercase; color: #16a34a; margin-bottom: 10px;
}
.hm-section-heading {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(1.8rem, 3vw, 2.4rem); font-weight: 800;
    color: #0f172a; margin-bottom: 10px; letter-spacing: -0.02em;
}
.hm-section-sub { color: #64748b; font-size: 1rem; line-height: 1.6; margin-bottom: 48px; max-width: 520px; }
.hm-feat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 28px; }
.hm-feat-card {
    background: white; border-radius: 22px; border: 1.5px solid #e2e8f0;
    padding: 32px 28px; box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    transition: transform 0.3s, box-shadow 0.3s;
    position: relative; overflow: hidden; cursor: default;
}
.hm-feat-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px;
}
.hm-feat-card.green::before { background: linear-gradient(90deg, #15803d, #22c55e, #86efac); }
.hm-feat-card.blue::before  { background: linear-gradient(90deg, #0284c7, #0ea5e9, #7dd3fc); }
.hm-feat-card.amber::before { background: linear-gradient(90deg, #b45309, #f59e0b, #fcd34d); }
/* Subtle glow behind card */
.hm-feat-card.green::after { content: ''; position: absolute; bottom: -40px; right: -40px; width: 120px; height: 120px; background: radial-gradient(circle, rgba(22,163,74,0.08) 0%, transparent 70%); border-radius: 50%; pointer-events: none; }
.hm-feat-card.blue::after  { content: ''; position: absolute; bottom: -40px; right: -40px; width: 120px; height: 120px; background: radial-gradient(circle, rgba(14,165,233,0.08) 0%, transparent 70%); border-radius: 50%; pointer-events: none; }
.hm-feat-card.amber::after { content: ''; position: absolute; bottom: -40px; right: -40px; width: 120px; height: 120px; background: radial-gradient(circle, rgba(245,158,11,0.08) 0%, transparent 70%); border-radius: 50%; pointer-events: none; }
.hm-feat-card:hover { transform: translateY(-8px); box-shadow: 0 24px 60px rgba(0,0,0,0.12); }
.hm-feat-icon {
    width: 54px; height: 54px; border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; margin-bottom: 20px;
    position: relative; z-index: 1;
}
.hm-feat-card.green .hm-feat-icon { background: linear-gradient(135deg, #dcfce7, #bbf7d0); box-shadow: 0 4px 14px rgba(22,163,74,0.25); }
.hm-feat-card.blue  .hm-feat-icon { background: linear-gradient(135deg, #dbeafe, #bae6fd); box-shadow: 0 4px 14px rgba(14,165,233,0.25); }
.hm-feat-card.amber .hm-feat-icon { background: linear-gradient(135deg, #fef9c3, #fde68a); box-shadow: 0 4px 14px rgba(245,158,11,0.25); }
.hm-feat-title { font-weight: 800; font-size: 1.08rem; color: #0f172a; margin-bottom: 12px; position: relative; z-index: 1; }
.hm-feat-desc { color: #64748b; font-size: 0.88rem; line-height: 1.7; margin-bottom: 20px; position: relative; z-index: 1; }
.hm-feat-tags { display: flex; flex-wrap: wrap; gap: 6px; position: relative; z-index: 1; }
.hm-feat-tag { font-size: 0.72rem; font-weight: 700; padding: 4px 11px; border-radius: 100px; }
.hm-feat-card.green .hm-feat-tag { background: #dcfce7; color: #15803d; }
.hm-feat-card.blue  .hm-feat-tag { background: #dbeafe; color: #0369a1; }
.hm-feat-card.amber .hm-feat-tag { background: #fef9c3; color: #92400e; }

/* ════════════════════════════════════════
   HOME — HOW IT WORKS
════════════════════════════════════════ */
.hm-how { padding: 80px 72px; background: white; }
.hm-steps {
    display: flex; align-items: flex-start; justify-content: center;
    gap: 0; margin-top: 48px;
}
.hm-step { flex: 1; text-align: center; padding: 0 24px; max-width: 240px; }
.hm-step-num {
    width: 64px; height: 64px;
    background: linear-gradient(135deg, #15803d, #22c55e);
    color: white; border-radius: 50%; display: flex; align-items: center;
    justify-content: center; font-family: 'Space Grotesk', sans-serif;
    font-size: 1.2rem; font-weight: 800; margin: 0 auto 18px;
    box-shadow: 0 8px 24px rgba(22,163,74,0.35), 0 0 0 8px rgba(22,163,74,0.1);
}
.hm-step-icon { font-size: 2rem; margin-bottom: 14px; display: block; }
.hm-step-title { font-weight: 800; font-size: 1.05rem; color: #0f172a; margin-bottom: 10px; }
.hm-step-desc { color: #64748b; font-size: 0.87rem; line-height: 1.65; }
.hm-step-connector {
    flex-shrink: 0; width: 80px; padding-top: 28px;
    display: flex; align-items: flex-start; justify-content: center; position: relative;
}
.hm-step-line {
    position: absolute; left: 10px; right: 10px; top: 60px;
    border-top: 2px dashed #bbf7d0;
}
.hm-step-arrow-txt {
    position: relative; z-index: 1; color: #22c55e;
    font-size: 1.4rem; background: white; padding: 0 6px; line-height: 1;
    margin-top: 24px;
}

/* ════════════════════════════════════════
   HOME — IN ACTION (example conversations)
════════════════════════════════════════ */
.hm-inaction { padding: 80px 72px; background: #f8fafc; }
.hm-action-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 24px; margin-top: 48px; }
.hm-action-card {
    background: white; border-radius: 20px; border: 1.5px solid #e2e8f0;
    padding: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    transition: transform 0.28s, box-shadow 0.28s;
    position: relative; overflow: hidden;
}
.hm-action-card:hover { transform: translateY(-5px); box-shadow: 0 16px 40px rgba(0,0,0,0.1); }
.hm-action-header {
    display: flex; align-items: center; gap: 10px; margin-bottom: 18px;
    padding-bottom: 14px; border-bottom: 1px solid #f1f5f9;
}
.hm-action-avatar {
    width: 38px; height: 38px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 1.1rem;
}
.hm-action-card:nth-child(1) .hm-action-avatar { background: linear-gradient(135deg, #dcfce7, #bbf7d0); }
.hm-action-card:nth-child(2) .hm-action-avatar { background: linear-gradient(135deg, #dbeafe, #bae6fd); }
.hm-action-card:nth-child(3) .hm-action-avatar { background: linear-gradient(135deg, #fef9c3, #fde68a); }
.hm-action-meta { flex: 1; }
.hm-action-topic { font-weight: 700; font-size: 0.88rem; color: #0f172a; }
.hm-action-lang  { font-size: 0.72rem; color: #94a3b8; margin-top: 1px; }
.hm-action-badge {
    font-size: 0.66rem; font-weight: 700; padding: 3px 9px;
    border-radius: 100px; text-transform: uppercase; letter-spacing: 0.06em;
}
.hm-action-card:nth-child(1) .hm-action-badge { background: #dcfce7; color: #15803d; }
.hm-action-card:nth-child(2) .hm-action-badge { background: #dbeafe; color: #0369a1; }
.hm-action-card:nth-child(3) .hm-action-badge { background: #fef9c3; color: #92400e; }
.hm-action-q {
    background: linear-gradient(135deg, #16a34a, #22c55e);
    color: white; border-radius: 14px 14px 14px 4px;
    padding: 10px 14px; font-size: 0.82rem; line-height: 1.5;
    margin-bottom: 10px; box-shadow: 0 3px 10px rgba(22,163,74,0.25);
}
.hm-action-a {
    background: #f8fafc; border: 1px solid #e2e8f0;
    color: #334155; border-radius: 14px 14px 4px 14px;
    padding: 10px 14px; font-size: 0.82rem; line-height: 1.5;
}
.hm-action-a strong { color: #16a34a; }

/* ════════════════════════════════════════
   HOME — LANGUAGES
════════════════════════════════════════ */
.hm-langs { padding: 80px 72px; background: white; }
.hm-lang-grid {
    display: grid; grid-template-columns: repeat(3,1fr); gap: 24px; margin-top: 48px;
}
.hm-lang-card {
    background: linear-gradient(145deg, #f0fdf4, #dcfce7);
    border: 1.5px solid #bbf7d0; border-radius: 22px;
    padding: 36px 24px; text-align: center;
    transition: transform 0.28s, box-shadow 0.28s;
    position: relative; overflow: hidden;
}
.hm-lang-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px;
    background: linear-gradient(90deg, #16a34a, #22c55e, #86efac);
}
.hm-lang-card:hover { transform: translateY(-6px); box-shadow: 0 12px 32px rgba(22,163,74,0.18); }
.hm-lang-flag { font-size: 2.6rem; margin-bottom: 14px; display: block; }
.hm-lang-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem; font-weight: 800; color: #15803d; margin-bottom: 8px;
}
.hm-lang-sub { color: #16a34a; font-size: 0.88rem; font-weight: 500; }

/* ════════════════════════════════════════
   HOME — CTA
════════════════════════════════════════ */
.hm-cta-section {
    background: linear-gradient(135deg, #020d06 0%, #052e16 30%, #14532d 60%, #166534 100%);
    padding: 96px 72px; text-align: center;
    position: relative; overflow: hidden;
}
.hm-cta-section::before {
    content: ''; position: absolute; inset: 0;
    background-image: radial-gradient(rgba(255,255,255,0.04) 1px, transparent 1px);
    background-size: 28px 28px; pointer-events: none;
}
.hm-cta-section::after {
    content: ''; position: absolute;
    top: 50%; left: 50%; transform: translate(-50%,-50%);
    width: 600px; height: 300px;
    background: radial-gradient(ellipse, rgba(34,197,94,0.15) 0%, transparent 65%);
    pointer-events: none;
}
.hm-cta-inner { position: relative; z-index: 1; }
.hm-cta-badge {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(34,197,94,0.15); border: 1px solid rgba(34,197,94,0.3);
    color: #86efac; font-size: 0.75rem; font-weight: 700;
    padding: 6px 16px; border-radius: 100px; margin-bottom: 24px;
    letter-spacing: 0.07em; text-transform: uppercase;
}
.hm-cta-heading {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(1.8rem, 3.5vw, 2.8rem); font-weight: 800;
    color: white; margin-bottom: 16px; letter-spacing: -0.02em;
}
.hm-cta-sub  { color: #86efac; font-size: 1.1rem; margin-bottom: 10px; }
.hm-cta-note { color: rgba(255,255,255,0.45); font-size: 0.82rem; margin-bottom: 36px; }

/* ════════════════════════════════════════
   NEW HOME DESIGN (nw-*)
════════════════════════════════════════ */
.nw-container { max-width: 1100px; margin: 0 auto; padding: 0 28px; }

/* ── Hero ── */
.nw-hero {
    position: relative; min-height: 100svh; overflow: hidden;
    background: linear-gradient(135deg, #f0fdf4 0%, #eff6ff 55%, #fefce8 100%);
    display: flex; align-items: center; justify-content: center;
}
.nw-bento-bg {
    position: absolute; inset: 0; padding: 14px; z-index: 0;
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    grid-template-rows: repeat(2, 1fr);
    gap: 10px;
}
.nw-cell { border-radius: 22px; overflow: hidden; box-shadow: 0 16px 48px rgba(0,0,0,0.12); }
.nw-cell:nth-child(1) { grid-column: span 1; animation: nwFdSc 0.9s ease 0.1s both; }
.nw-cell:nth-child(2) { grid-column: span 2; animation: nwFdSc 0.9s ease 0.2s both; }
.nw-cell:nth-child(3) { grid-column: span 2; animation: nwFdSc 0.9s ease 0.3s both; }
.nw-cell:nth-child(4) { grid-column: span 1; animation: nwFdSc 0.9s ease 0.4s both; }
@keyframes nwFdSc { from { opacity:0; transform:scale(0.88); } to { opacity:1; transform:scale(1); } }
.nw-cell img { width:100%; height:100%; object-fit:cover; transition:transform 0.6s ease; }
.nw-cell:hover img { transform:scale(1.05); }
.nw-hero-content {
    position: relative; z-index: 10; text-align: center;
    max-width: 660px; width: 90%;
    padding: 44px 40px;
    background: rgba(255,255,255,0.9);
    backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
    border-radius: 28px;
    box-shadow: 0 20px 72px rgba(0,0,0,0.1), 0 4px 16px rgba(0,0,0,0.05);
    border: 1.5px solid rgba(255,255,255,0.95);
    animation: nwSlUp 0.9s ease 0.55s both;
}
@keyframes nwSlUp { from { opacity:0; transform:translateY(26px) scale(0.97); } to { opacity:1; transform:translateY(0) scale(1); } }
.nw-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 16px; border-radius: 100px;
    font-size: 0.79rem; font-weight: 600; margin-bottom: 18px;
    font-family: 'Plus Jakarta Sans', sans-serif; letter-spacing: 0.01em;
}
.nw-badge-green  { background:#dcfce7; color:#15803d; border:1px solid #bbf7d0; }
.nw-badge-blue   { background:#dbeafe; color:#1d4ed8; border:1px solid #bfdbfe; }
.nw-badge-yellow { background:#fef9c3; color:#854d0e; border:1px solid #fef08a; }
.nw-badge-white  { background:rgba(255,255,255,0.2); color:white; border:1px solid rgba(255,255,255,0.35); }
.nw-h1 {
    font-family:'Space Grotesk',sans-serif;
    font-size:clamp(2.2rem,5vw,3.6rem); font-weight:800;
    color:#0f172a; line-height:1.15; margin-bottom:14px; letter-spacing:-0.03em;
}
.nw-gradient-text {
    background:linear-gradient(90deg,#16a34a,#2563eb,#ca8a04);
    background-size:300% auto;
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    animation:nwGSh 5s linear infinite;
}
@keyframes nwGSh { 0%{background-position:0%} 100%{background-position:300%} }
.nw-hero-desc { color:#475569; font-size:1rem; line-height:1.75; margin-bottom:26px; }
.nw-hero-btns { display:flex; gap:10px; justify-content:center; flex-wrap:wrap; }
.nw-btn-primary {
    display:inline-flex; align-items:center; gap:7px;
    background:#16a34a; color:white; border:none; border-radius:100px;
    padding:13px 26px; font-size:0.94rem; font-weight:700;
    font-family:'Plus Jakarta Sans',sans-serif; cursor:pointer;
    transition:all 0.25s; box-shadow:0 6px 20px rgba(22,163,74,0.4);
}
.nw-btn-primary:hover { background:#15803d; transform:translateY(-2px); box-shadow:0 10px 30px rgba(22,163,74,0.5); }
.nw-btn-outline {
    display:inline-flex; align-items:center; gap:7px;
    background:transparent; color:#16a34a; border:2px solid #16a34a;
    border-radius:100px; padding:13px 26px; font-size:0.94rem; font-weight:700;
    font-family:'Plus Jakarta Sans',sans-serif; cursor:pointer; transition:all 0.25s;
}
.nw-btn-outline:hover { background:#f0fdf4; transform:translateY(-2px); }

/* ── Hide the real Gradio buttons (keep clickable via JS) ── */
#km-home-cta-row {
    visibility: hidden !important;
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: visible !important;
}

/* ── Features ── */
.nw-features { padding:56px 0; background:rgba(255,255,255,0.72); backdrop-filter:blur(8px); }
.nw-section-head { text-align:center; margin-bottom:36px; }
.nw-section-title {
    font-family:'Space Grotesk',sans-serif; font-size:clamp(1.75rem,3.5vw,2.5rem);
    font-weight:800; color:#0f172a; margin-bottom:10px; letter-spacing:-0.02em;
}
.nw-section-desc { color:#64748b; font-size:0.97rem; max-width:720px; margin:0 auto; line-height:1.7; }
.nw-feat-grid {
    display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
    gap:22px; max-width:1060px; margin:0 auto;
}

/* Action Cards */
.nw-act-card {
    background:white; border:1.5px solid #e2e8f0; border-radius:20px;
    padding:26px 24px 20px; box-shadow:0 4px 20px rgba(0,0,0,0.06);
    position:relative; overflow:hidden; cursor:default;
    transition:transform 0.28s,box-shadow 0.28s,border-color 0.28s;
    display:flex; flex-direction:column;
    animation:nwActIn 0.55s ease both;
}
/* Staggered entrance per card position */
.nw-act-card:nth-child(1){animation-delay:0.05s}
.nw-act-card:nth-child(2){animation-delay:0.12s}
.nw-act-card:nth-child(3){animation-delay:0.19s}
.nw-act-card:nth-child(4){animation-delay:0.26s}
.nw-act-card:nth-child(5){animation-delay:0.33s}
.nw-act-card:nth-child(6){animation-delay:0.40s}
@keyframes nwActIn{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
.nw-act-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:4px;
}
/* Top border gradient per colour */
.nw-ac-green::before  { background:linear-gradient(90deg,#15803d,#22c55e,#86efac); }
.nw-ac-blue::before   { background:linear-gradient(90deg,#1d4ed8,#3b82f6,#93c5fd); }
.nw-ac-yellow::before { background:linear-gradient(90deg,#b45309,#f59e0b,#fcd34d); }
.nw-ac-purple::before { background:linear-gradient(90deg,#7c3aed,#a855f7,#d8b4fe); }
.nw-ac-red::before    { background:linear-gradient(90deg,#b91c1c,#ef4444,#fca5a5); }
.nw-ac-indigo::before { background:linear-gradient(90deg,#3730a3,#6366f1,#a5b4fc); }
/* Hover lift + coloured border */
.nw-act-card:hover { transform:translateY(-7px); box-shadow:0 24px 64px rgba(0,0,0,0.12); }
.nw-ac-green:hover  { border-color:#bbf7d0; }
.nw-ac-blue:hover   { border-color:#bfdbfe; }
.nw-ac-yellow:hover { border-color:#fde68a; }
.nw-ac-purple:hover { border-color:#e9d5ff; }
.nw-ac-red:hover    { border-color:#fecaca; }
.nw-ac-indigo:hover { border-color:#c7d2fe; }
/* Card header row */
.nw-ac-head { display:flex; align-items:center; justify-content:space-between; margin-bottom:16px; }
/* Icon box */
.nw-ac-icon {
    width:50px; height:50px; border-radius:14px;
    display:flex; align-items:center; justify-content:center; font-size:1.35rem;
}
.nw-ac-green  .nw-ac-icon { background:linear-gradient(135deg,#dcfce7,#bbf7d0); box-shadow:0 4px 14px rgba(22,163,74,0.2); }
.nw-ac-blue   .nw-ac-icon { background:linear-gradient(135deg,#dbeafe,#bfdbfe); box-shadow:0 4px 14px rgba(59,130,246,0.2); }
.nw-ac-yellow .nw-ac-icon { background:linear-gradient(135deg,#fef9c3,#fde68a); box-shadow:0 4px 14px rgba(245,158,11,0.2); }
.nw-ac-purple .nw-ac-icon { background:linear-gradient(135deg,#f3e8ff,#e9d5ff); box-shadow:0 4px 14px rgba(168,85,247,0.2); }
.nw-ac-red    .nw-ac-icon { background:linear-gradient(135deg,#fef2f2,#fecaca); box-shadow:0 4px 14px rgba(239,68,68,0.2); }
.nw-ac-indigo .nw-ac-icon { background:linear-gradient(135deg,#eef2ff,#c7d2fe); box-shadow:0 4px 14px rgba(99,102,241,0.2); }
/* Status badge */
.nw-ac-tag {
    font-size:0.63rem; font-weight:800; padding:3px 10px;
    border-radius:100px; text-transform:uppercase; letter-spacing:0.06em;
}
.nw-ac-green  .nw-ac-tag { background:#dcfce7; color:#15803d; }
.nw-ac-blue   .nw-ac-tag { background:#dbeafe; color:#1d4ed8; }
.nw-ac-yellow .nw-ac-tag { background:#fef9c3; color:#854d0e; }
.nw-ac-purple .nw-ac-tag { background:#f3e8ff; color:#7c3aed; }
.nw-ac-red    .nw-ac-tag { background:#fef2f2; color:#b91c1c; }
.nw-ac-indigo .nw-ac-tag { background:#eef2ff; color:#3730a3; }
/* Title + body */
.nw-ac-title { font-family:'Space Grotesk',sans-serif; font-weight:800; font-size:1.05rem; color:#0f172a; margin-bottom:8px; }
.nw-ac-desc  { color:#64748b; font-size:0.85rem; line-height:1.7; margin-bottom:16px; flex:1; }
/* Capability chips */
.nw-ac-chips { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:16px; }
.nw-ac-chip  {
    font-size:0.71rem; font-weight:600; padding:4px 10px;
    border-radius:100px; background:#f8fafc; border:1px solid #e2e8f0; color:#475569;
}
/* Card footer link */
.nw-ac-footer {
    border-top:1px solid #f1f5f9; padding-top:13px;
    font-size:0.82rem; font-weight:700;
    display:flex; align-items:center; justify-content:space-between;
    transition:gap 0.2s;
}
.nw-act-card:hover .nw-ac-footer { gap:6px; }
.nw-ac-green  .nw-ac-footer { color:#15803d; }
.nw-ac-blue   .nw-ac-footer { color:#1d4ed8; }
.nw-ac-yellow .nw-ac-footer { color:#854d0e; }
.nw-ac-purple .nw-ac-footer { color:#7c3aed; }
.nw-ac-red    .nw-ac-footer { color:#b91c1c; }
.nw-ac-indigo .nw-ac-footer { color:#3730a3; }

/* ── Stats ── */
.nw-stats {
    padding:84px 0;
    background:linear-gradient(135deg,#16a34a 0%,#15803d 30%,#1d4ed8 70%,#1e40af 100%);
    color:white; position:relative; overflow:hidden;
}
.nw-stats::before {
    content:''; position:absolute; inset:0;
    background-image:radial-gradient(rgba(255,255,255,0.05) 1px,transparent 1px);
    background-size:26px 26px; pointer-events:none;
}
.nw-stats-inner { position:relative; z-index:1; }
.nw-title-white { color:white !important; }
.nw-desc-light  { color:rgba(187,247,208,0.85) !important; }
.nw-stats-grid {
    display:grid; grid-template-columns:repeat(auto-fit,minmax(196px,1fr));
    gap:18px; max-width:940px; margin:0 auto 44px;
}
.nw-stat-card {
    background:rgba(255,255,255,0.12); backdrop-filter:blur(10px);
    border:1px solid rgba(255,255,255,0.2); border-radius:20px;
    padding:28px 18px; text-align:center; transition:background 0.3s;
    opacity:0; transform:translateY(18px);
}
.nw-stat-card.nw-in { opacity:1; transform:translateY(0); transition:opacity 0.5s ease,transform 0.5s ease; }
.nw-stat-card:hover { background:rgba(255,255,255,0.2); }
.nw-stat-icon { font-size:1.75rem; margin-bottom:8px; }
.nw-stat-num {
    font-family:'Space Grotesk',sans-serif; font-size:2.2rem; font-weight:800;
    color:white; margin-bottom:5px; display:flex; align-items:center; justify-content:center;
}
.nw-stat-label { color:rgba(187,247,208,0.85); font-size:0.9rem; }
.nw-trust-items { display:flex; gap:28px; justify-content:center; flex-wrap:wrap; margin-bottom:28px; }
.nw-trust-item  { font-size:0.92rem; color:rgba(255,255,255,0.9); }
.nw-stats-cta   { text-align:center; }
.nw-btn-white {
    display:inline-flex; align-items:center; gap:7px;
    background:white; color:#16a34a; border:none; border-radius:100px;
    padding:14px 34px; font-size:0.97rem; font-weight:700;
    font-family:'Plus Jakarta Sans',sans-serif; cursor:pointer;
    transition:all 0.25s; box-shadow:0 6px 20px rgba(0,0,0,0.13);
}
.nw-btn-white:hover { background:#f0fdf4; transform:translateY(-2px); box-shadow:0 12px 34px rgba(0,0,0,0.18); }

/* ── Testimonials ── */
.nw-testimonials { padding:84px 0; background:rgba(255,255,255,0.72); backdrop-filter:blur(8px); }
.nw-test-grid {
    display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
    gap:20px; max-width:1060px; margin:0 auto;
}
.nw-test-card {
    background:white; border:2px solid #bbf7d0; border-radius:20px;
    padding:26px 24px; box-shadow:0 4px 20px rgba(0,0,0,0.05);
    transition:all 0.3s; opacity:0; transform:translateY(18px);
}
.nw-test-card.nw-in { opacity:1; transform:translateY(0); transition:opacity 0.5s ease,transform 0.5s ease,box-shadow 0.3s,border-color 0.3s; }
.nw-test-card:hover { border-color:#86efac; box-shadow:0 14px 44px rgba(22,163,74,0.1); transform:translateY(-4px); }
.nw-stars  { color:#eab308; font-size:1.05rem; margin-bottom:10px; letter-spacing:2px; }
.nw-test-quote { color:#334155; font-size:0.92rem; line-height:1.72; font-style:italic; margin-bottom:16px; }
.nw-test-author{ display:flex; align-items:center; gap:10px; }
.nw-test-avatar{ font-size:2.1rem; }
.nw-test-name  { font-weight:700; font-size:0.9rem; color:#0f172a; margin-bottom:1px; }
.nw-test-role  { color:#64748b; font-size:0.76rem; }

/* ── CTA ── */
.nw-cta-section {
    padding:52px 24px;
    background:linear-gradient(135deg,#f0fdf4,#eff6ff,#fefce8);
    display:flex; align-items:center; justify-content:center;
}
.nw-cta-card {
    background:white; border:3px solid #bbf7d0; border-radius:28px;
    padding:52px 56px; text-align:center; max-width:740px; width:100%;
    box-shadow:0 20px 72px rgba(0,0,0,0.09);
}
.nw-cta-icon  { font-size:3rem; margin-bottom:16px; }
.nw-cta-title {
    font-family:'Space Grotesk',sans-serif; font-size:clamp(1.65rem,3.5vw,2.4rem);
    font-weight:800; color:#0f172a; margin-bottom:12px; letter-spacing:-0.02em;
}
.nw-cta-desc  { color:#64748b; font-size:0.97rem; line-height:1.7; max-width:480px; margin:0 auto 26px; }
.nw-btn-gradient {
    display:inline-flex; align-items:center; gap:7px;
    background:linear-gradient(135deg,#16a34a,#2563eb); color:white; border:none;
    border-radius:100px; padding:14px 34px; font-size:0.97rem; font-weight:700;
    font-family:'Plus Jakarta Sans',sans-serif; cursor:pointer;
    transition:all 0.25s; box-shadow:0 7px 24px rgba(22,163,74,0.38);
}
.nw-btn-gradient:hover { transform:translateY(-2px); box-shadow:0 12px 36px rgba(22,163,74,0.48); filter:brightness(1.07); }
.nw-cta-note  { color:#94a3b8; font-size:0.8rem; margin-top:10px; }

/* ════════════════════════════════════════
   ADVISOR — LAYOUT WRAPPER
════════════════════════════════════════ */
#km-split-row {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: stretch !important;
    gap: 20px !important;
    padding: 24px !important;
    min-height: calc(100vh - 68px) !important;
    background: linear-gradient(135deg, #f0fdf4 0%, #eff6ff 50%, #f0fdf4 100%) !important;
    background-size: 400% 400% !important;
    animation: advisorBg 20s ease infinite !important;
    position: relative !important;
    z-index: 1 !important;
    box-sizing: border-box !important;
}

/* Gradio wraps each Column in an extra div — target it */
#km-split-row > div {
    display: flex !important;
    flex-direction: column !important;
    min-width: 0 !important;
    height: auto !important;
}

/* Responsive: Stack on smaller screens */
@media (max-width: 900px) {
    #km-split-row {
        flex-direction: column !important;
        flex-wrap: wrap !important;
        min-height: auto !important;
    }
    #km-split-row > div {
        width: 100% !important;
        flex: none !important;
    }
}

/* ════════════════════════════════════════
   ADVISOR — SIDEBAR CARD
════════════════════════════════════════ */
#km-sidebar {
    background: white !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 22px !important;
    box-shadow: 0 8px 40px rgba(0,0,0,0.09), 0 2px 10px rgba(0,0,0,0.04) !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    flex: 0 0 340px !important;
    width: 340px !important;
    min-width: 300px !important;
    max-width: 380px !important;
    max-height: calc(100vh - 92px) !important;
    position: sticky !important;
    top: 24px !important;
}

/* Sidebar scrollbar styling */
#km-sidebar::-webkit-scrollbar {
    width: 5px;
}

#km-sidebar::-webkit-scrollbar-track {
    background: transparent;
}

#km-sidebar::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 3px;
}

#km-sidebar::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
}

/* ════════════════════════════════════════
   EXAMPLE QUERIES
════════════════════════════════════════ */

/* Accordion wrapper — card feel */
.km-accord {
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 16px !important;
    overflow: hidden !important;
    margin-top: 12px !important;
    background: white !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04) !important;
}

/* Accordion chevron + label area */
.km-accord .label,
.km-accord .label-wrap {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    color: #166534 !important;
    padding: 14px 18px !important;
    gap: 10px !important;
    align-items: center !important;
    letter-spacing: 0.01em !important;
}

/* Accordion icon (chevron) */
.km-accord .label .icon { color: #166534 !important; }

/* Table inside examples */
.km-accord .examples {
    padding: 4px 0 6px !important;
}

.km-accord .examples > table {
    width: 100% !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
}

/* Table header — column labels */
.km-accord .examples > table thead th {
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: #94a3b8 !important;
    padding: 10px 18px 6px !important;
    text-align: left !important;
    border-bottom: 1px solid #f1f5f9 !important;
    background: transparent !important;
}
.km-accord .examples > table thead th:last-child {
    text-align: right !important;
}

/* Table body rows */
.km-accord .examples > table tbody tr {
    cursor: pointer !important;
    transition: background 0.18s !important;
}
.km-accord .examples > table tbody tr:hover {
    background: #f0fdf4 !important;
}

/* Table cells */
.km-accord .examples > table tbody td {
    font-size: 0.9rem !important;
    color: #334155 !important;
    padding: 12px 18px !important;
    line-height: 1.45 !important;
    border-bottom: 1px solid #f1f5f9 !important;
    background: transparent !important;
}

/* Language column (rightmost) */
.km-accord .examples > table tbody td:last-child {
    text-align: right !important;
    width: 80px !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    color: #1e40af !important;
    letter-spacing: 0.02em !important;
}
.km-accord .examples > table tbody td:last-child .text,
.km-accord .examples > table tbody td:last-child * {
    display: inline-block !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    padding: 3px 10px !important;
    border-radius: 100px !important;
    color: #1e40af !important;
    background: #eff6ff !important;
    border: 1px solid #dbeafe !important;
    white-space: nowrap !important;
}

/* First column takes remaining width */
.km-accord .examples > table tbody td:first-child {
    width: auto !important;
}

/* Remove any lingering alternating row colors from Gradio defaults */
.km-accord .examples > table tbody tr:nth-child(even),
.km-accord .examples > table tbody tr:nth-child(odd) {
    background: transparent !important;
}

/* Scrollbar for the examples container if needed */
.km-accord .examples { overflow-x: auto !important; }



/* Hide Gradio's per-component clear/reset "X" icons in sidebar */
#km-sidebar button[aria-label*="Clear"],
#km-sidebar button[title*="Clear"],
#km-sidebar button[data-testid*="clear"],
#km-sidebar .icon-button[aria-label*="Remove"] {
    display: none !important;
}

/* Image upload — visible drop zone */
#km-image-in {
    border: 2px dashed #d1d5db !important;
    border-radius: 12px !important;
    padding: 0 !important;
    background: white !important;
    min-height: 120px !important;
    overflow: hidden !important;
}

/* Ensure the Gradio upload button/icon/text inside the image component is visible */
#km-image-in button[class*="upload"],
#km-image-in [class*="upload-container"] button {
    background: transparent !important;
    color: #15803d !important;
    border: none !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}
#km-image-in [class*="upload-container"] {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 100px !important;
    color: #475569 !important;
}

/* Labels */
#km-audio-in label,
#km-image-in label,
#km-lang label,
#km-audio-out label {
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    color: #475569 !important;
}

/* Audio / Image buttons */
#km-audio-in button,
#km-image-in button {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7) !important;
    border: 1.5px solid #bbf7d0 !important;
    color: #15803d !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 8px 14px !important;
    font-size: 0.82rem !important;
    transition: all 0.2s !important;
    min-height: 36px !important;
}
#km-audio-in button:hover,
#km-image-in button:hover {
    background: linear-gradient(135deg, #bbf7d0, #86efac) !important;
    box-shadow: 0 2px 8px rgba(22,163,74,0.15) !important;
}

/* Language dropdown */
#km-lang select, #km-lang input {
    border-radius: 10px !important;
    border: 1.5px solid #e2e8f0 !important;
    padding: 8px 12px !important;
    min-height: 38px !important;
    font-size: 0.85rem !important;
}

/* Clear button */
#km-clear button {
    width: 100% !important;
    background: white !important;
    color: #64748b !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 10px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    transition: all 0.22s !important;
    cursor: pointer !important;
}
#km-clear button:hover {
    background: #fff5f5 !important;
    color: #dc2626 !important;
    border-color: #fca5a5 !important;
    box-shadow: 0 2px 8px rgba(239,68,68,0.12) !important;
}

/* Separator rule */
.km-sb-rule {
    border: none !important;
    border-top: 1px solid #e2e8f0 !important;
    margin: 0 12px !important;
}

/* Sidebar dark header */
.km-sb-head {
    background: linear-gradient(145deg, #052e16 0%, #0a3d1e 30%, #14532d 65%, #166534 100%);
    padding: 18px 18px 14px;
    position: relative;
    overflow: hidden;
    flex-shrink: 0;
    border-bottom: 1px solid rgba(22,163,74,0.2);
}
.km-sb-head::before {
    content: '';
    position: absolute;
    inset: 0;
    background-image: radial-gradient(rgba(255,255,255,0.055) 1px, transparent 1px);
    background-size: 18px 18px;
    pointer-events: none;
}
.km-sb-head::after {
    content: '';
    position: absolute;
    top: -40px;
    right: -40px;
    width: 150px;
    height: 150px;
    background: radial-gradient(circle, rgba(74,222,128,0.2) 0%, transparent 65%);
    border-radius: 50%;
    pointer-events: none;
}
.km-sb-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    position: relative;
    z-index: 1;
    margin-bottom: 10px;
}
.km-sb-logo-icon {
    width: 40px;
    height: 40px;
    background: rgba(255,255,255,0.15);
    border-radius: 11px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.2);
}
.km-sb-logo-text {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 800;
    color: white;
    font-size: 1.05rem;
    line-height: 1;
}
.km-sb-logo-sub {
    color: #86efac;
    font-size: 0.7rem;
    font-weight: 500;
    opacity: 0.85;
    margin-top: 2px;
}
.km-sb-status {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(34,197,94,0.15);
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 100px;
    padding: 5px 12px 5px 9px;
    font-size: 0.72rem;
    font-weight: 600;
    color: #86efac;
    position: relative;
    z-index: 1;
    width: fit-content;
}
.km-sb-status-dot {
    width: 6px;
    height: 6px;
    background: #4ade80;
    border-radius: 50%;
    animation: pulseDot 2s ease-in-out infinite;
    flex-shrink: 0;
}

/* Sidebar capabilities strip */
.km-sb-caps {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    padding: 10px 14px;
    background: white;
    border-bottom: 1px solid #f1f5f9;
    flex-shrink: 0;
}
.km-sb-cap {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    color: #475569;
    background: linear-gradient(135deg, #f8fafc, #f1f5f9);
    border: 1px solid #e2e8f0;
    padding: 4px 10px;
    border-radius: 100px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    transition: all 0.2s;
}
.km-sb-cap:hover {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border-color: #86efac;
    color: #15803d;
}

/* Sidebar rule separator */
.km-sb-rule {
    border: none !important;
    border-top: 1px solid #e2e8f0 !important;
    margin: 2px 16px !important;
}

/* Component labels */
#km-audio-in label,
#km-image-in label,
#km-lang label,
#km-audio-out label {
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    color: #475569 !important;
    padding: 0 0 4px 0 !important;
    margin: 0 !important;
}

/* Audio/Image upload buttons */
#km-audio-in button,
#km-image-in button {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7) !important;
    border: 1.5px solid #bbf7d0 !important;
    color: #15803d !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 8px 14px !important;
    font-size: 0.82rem !important;
    transition: all 0.2s !important;
    min-height: 36px !important;
}
#km-audio-in button:hover,
#km-image-in button:hover {
    background: linear-gradient(135deg, #bbf7d0, #86efac) !important;
    box-shadow: 0 2px 8px rgba(22,163,74,0.15) !important;
}

/* Language dropdown */
#km-lang select, #km-lang input {
    border-radius: 10px !important;
    border: 1.5px solid #e2e8f0 !important;
    padding: 8px 12px !important;
    min-height: 38px !important;
    font-size: 0.85rem !important;
}

/* Clear button */
#km-clear button {
    width: 100% !important;
    background: white !important;
    color: #64748b !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 10px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    transition: all 0.22s !important;
    cursor: pointer !important;
}
#km-clear button:hover {
    background: #fff5f5 !important;
    color: #dc2626 !important;
    border-color: #fca5a5 !important;
    box-shadow: 0 2px 8px rgba(239,68,68,0.12) !important;
}

/* ════════════════════════════════════════
   ADVISOR — CHAT CARD
════════════════════════════════════════ */
#km-chat-col {
    background: white !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 22px !important;
    box-shadow: 0 8px 40px rgba(0,0,0,0.09), 0 2px 10px rgba(0,0,0,0.04) !important;
    overflow: hidden !important;
    padding: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    flex: 1 1 auto !important;
    min-width: 0 !important;
    /* Don't force a height — let chatbot's fixed height drive the size */
}

/* Chat column inner Gradio wrapper — zero out chrome */
#km-chat-col > div,
#km-chat-col > div > div {
    padding: 0 !important;
    gap: 0 !important;
    width: 100% !important;
}

/* Responsive: Adjust on smaller screens */
@media (max-width: 900px) {
    #km-chat-col {
        min-height: 500px !important;
    }
}

/* Chat header — compact */
.km-chat-head {
    padding: 16px 18px 12px;
    border-bottom: 2px solid #f1f5f9;
    background: linear-gradient(135deg, #f0fdf4, #ffffff);
    display: flex;
    align-items: center;
    gap: 12px;
    flex-shrink: 0;
}

.km-chat-head-icon {
    width: 48px;
    height: 48px;
    border-radius: 14px;
    flex-shrink: 0;
    background: linear-gradient(135deg, #14532d, #22c55e);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.35rem;
    box-shadow: 0 6px 16px rgba(22,163,74,0.35);
}

.km-chat-head-title {
    font-weight: 800;
    font-size: 1.08rem;
    color: #0f172a;
}

.km-chat-head-sub {
    color: #64748b;
    font-size: 0.75rem;
    margin-top: 2px;
    font-weight: 500;
}

.km-ai-badge {
    margin-left: auto;
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    color: #16a34a;
    border: 1px solid #bbf7d0;
    font-size: 0.73rem;
    font-weight: 700;
    padding: 6px 14px;
    border-radius: 100px;
    display: flex;
    align-items: center;
    gap: 6px;
    box-shadow: 0 2px 8px rgba(22,163,74,0.15);
}

.km-ai-dot {
    width: 7px;
    height: 7px;
    background: #22c55e;
    border-radius: 50%;
    animation: pulseDot 2s ease-in-out infinite;
}

/* "Try:" label bar above quick buttons */
.km-quick-bar {
    padding: 8px 14px 0 !important;
    background: #fafcff;
    font-size: 0.68rem;
    font-weight: 700;
    color: #94a3b8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    flex-shrink: 0;
    border-bottom: 1px solid #f1f5f9;
}

/* Quick prompts row — horizontal pill chips */
#km-quick-row {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    gap: 8px !important;
    padding: 10px 14px !important;
    background: #fafcff !important;
    border-bottom: 1px solid #f1f5f9 !important;
    overflow-x: auto !important;
    flex-shrink: 0 !important;
    min-height: unset !important;
    height: auto !important;
}

#km-quick-row::-webkit-scrollbar { height: 3px; }
#km-quick-row::-webkit-scrollbar-track { background: transparent; }
#km-quick-row::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 10px; }

/* Button wrapper — don't stretch, shrink to content */
#km-quick-row > div {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: unset !important;
}

/* Quick prompt pill buttons */
#km-q1 button, #km-q2 button, #km-q3 button {
    background: white !important;
    border: 1.5px solid #e2e8f0 !important;
    color: #475569 !important;
    border-radius: 100px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    padding: 6px 14px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
    transition: all 0.22s !important;
    white-space: nowrap !important;
    min-height: unset !important;
    height: 32px !important;
    line-height: 1 !important;
    cursor: pointer !important;
    display: inline-flex !important;
    align-items: center !important;
}

#km-q1 button:hover { background:#dcfce7 !important; color:#15803d !important; border-color:#86efac !important; box-shadow:0 3px 10px rgba(22,163,74,0.15) !important; }
#km-q2 button:hover { background:#dbeafe !important; color:#1d4ed8 !important; border-color:#93c5fd !important; box-shadow:0 3px 10px rgba(29,78,216,0.15) !important; }
#km-q3 button:hover { background:#fef9c3 !important; color:#92400e !important; border-color:#fcd34d !important; box-shadow:0 3px 10px rgba(180,83,9,0.15) !important; }

#km-q1, #km-q2, #km-q3 { flex-shrink: 0 !important; }

/* Chatbot — takes all remaining vertical space */
#km-chatbot {
    background: #f8fafc !important;
    border: none !important;
    flex: 1 1 auto !important;
    min-height: 250px !important;
    max-height: calc(100vh - 400px) !important;
    overflow-y: auto !important;
}

#km-chatbot > div {
    border: none !important;
    background: transparent !important;
    flex: 1 1 auto !important;
    min-height: 0 !important;
}

#km-chatbot ::-webkit-scrollbar {
    width: 6px;
}

#km-chatbot ::-webkit-scrollbar-track {
    background: transparent;
}

#km-chatbot ::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 10px;
}

#km-chatbot ::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
}

/* Input row — always visible at bottom */
#km-input-row {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 10px !important;
    padding: 10px 14px !important;
    border-top: 2px solid #f1f5f9 !important;
    background: white !important;
    align-items: flex-end !important;
    flex-shrink: 0 !important;
    flex-grow: 0 !important;
    border-radius: 0 0 22px 22px !important;
    min-height: 60px !important;
    max-height: 110px !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

/* Textbox wrapper takes all available width */
#km-input-row > div:first-child {
    flex: 1 1 auto !important;
    min-width: 0 !important;
    width: auto !important;
}

/* Send button wrapper stays fixed */
#km-input-row > div:last-child {
    flex: 0 0 auto !important;
    min-width: 100px !important;
    width: auto !important;
}

#km-textbox {
    min-width: 0 !important;
    width: 100% !important;
}

#km-textbox label span {
    display: none !important;
}

#km-textbox textarea {
    border-radius: 14px !important;
    border: 1.5px solid #e2e8f0 !important;
    padding: 12px 16px !important;
    resize: vertical !important;
    max-height: 100px !important;
    min-height: 42px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.9rem !important;
    background: #f8fafc !important;
    transition: all 0.2s !important;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.04) !important;
}

#km-textbox textarea:focus {
    border-color: #16a34a !important;
    background: white !important;
    box-shadow: 0 0 0 3px rgba(22,163,74,0.12), inset 0 1px 2px rgba(0,0,0,0.04) !important;
    outline: none !important;
}

#km-textbox textarea::placeholder {
    color: #a0aec0 !important;
}

/* Send button */
#km-send {
    display: flex !important;
    flex-direction: column !important;
    flex-shrink: 0 !important;
    min-width: 110px !important;
    gap: 0 !important;
}

#km-send button {
    flex: 1 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 6px !important;
    background: linear-gradient(135deg, #16a34a 0%, #15803d 40%, #1d4ed8 100%) !important;
    color: #fff !important;
    font-weight: 800 !important;
    font-size: 0.92rem !important;
    border-radius: 12px !important;
    border: none !important;
    min-height: 45px !important;
    width: 100% !important;
    opacity: 1 !important;
    visibility: visible !important;
    cursor: pointer !important;
    letter-spacing: 0.02em !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    box-shadow: 0 6px 20px rgba(22,163,74,0.45) !important;
    animation: btnGlow 3s ease-in-out infinite !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
    position: relative !important;
    overflow: hidden !important;
    padding: 0 16px !important;
}

#km-send button::after {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: -120% !important;
    width: 60% !important;
    height: 100% !important;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.25), transparent) !important;
    transition: left 0.55s ease !important;
    pointer-events: none !important;
}

#km-send button:hover::after {
    left: 170% !important;
}

#km-send button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 32px rgba(22,163,74,0.6) !important;
}

#km-send button:active {
    transform: translateY(0) scale(0.97) !important;
}

/* ════════════════════════════════════════
   ACCORDION & FOOTER
════════════════════════════════════════ */
.km-accord {
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 14px !important;
    margin: 8px 14px 10px !important;
    overflow: hidden !important;
    background: white !important;
    flex-shrink: 0 !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05) !important;
    transition: all 0.2s !important;
}

.km-accord:hover {
    border-color: #bbf7d0 !important;
    box-shadow: 0 4px 12px rgba(22,163,74,0.08) !important;
}

.km-footer {
    text-align: center;
    padding: 14px 18px;
    color: #64748b;
    font-size: 0.76rem;
    font-weight: 500;
    border-top: 1px solid #f1f5f9;
    background: white;
    border-radius: 0 0 22px 22px;
    flex-shrink: 0;
    letter-spacing: 0.01em;
}

/* ════════════════════════════════════════
   ANIMATIONS
════════════════════════════════════════ */
/* ════════════════════════════════════════
   ANIMATED BACKGROUND ORBS (both tabs)
════════════════════════════════════════ */
.gradio-container::after {
    content: '';
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(circle at 12% 20%, rgba(22,163,74,0.07) 0%, transparent 32%),
        radial-gradient(circle at 88% 15%, rgba(14,165,233,0.05) 0%, transparent 28%),
        radial-gradient(circle at 55% 80%, rgba(168,85,247,0.04) 0%, transparent 26%),
        radial-gradient(circle at 25% 65%, rgba(251,191,36,0.04) 0%, transparent 22%);
    pointer-events: none;
    z-index: 0;
    animation: bgOrbs 30s ease-in-out infinite;
}
@keyframes bgOrbs {
    0%, 100% { transform: translate(0, 0) scale(1); }
    25%  { transform: translate(20px, -20px) scale(1.03); }
    50%  { transform: translate(-15px, 15px) scale(0.97); }
    75%  { transform: translate(10px, 25px) scale(1.02); }
}

/* ════════════════════════════════════════
   HOME — BENTO FEATURE GRID
════════════════════════════════════════ */
.hm-bento {
    display: grid;
    grid-template-columns: 5fr 7fr;
    grid-template-rows: auto auto;
    gap: 22px;
    margin-top: 48px;
}
.hm-bc-main { grid-row: span 2; }

.hm-bento-card {
    background: white; border-radius: 24px;
    border: 1.5px solid #e2e8f0;
    padding: 30px 28px 26px;
    overflow: hidden; position: relative;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
    cursor: default;
}
.hm-bento-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px;
}
.hm-bento-card.bc-green::before { background: linear-gradient(90deg, #15803d, #22c55e, #86efac); }
.hm-bento-card.bc-blue::before  { background: linear-gradient(90deg, #0284c7, #0ea5e9, #7dd3fc); }
.hm-bento-card.bc-amber::before { background: linear-gradient(90deg, #b45309, #f59e0b, #fcd34d); }
.hm-bento-card:hover { transform: translateY(-6px); box-shadow: 0 20px 56px rgba(0,0,0,0.11); }

.hm-bc-badge {
    position: absolute; top: 22px; right: 22px;
    font-size: 0.65rem; font-weight: 800; padding: 3px 10px;
    border-radius: 100px; text-transform: uppercase; letter-spacing: 0.07em;
}
.hm-bc-badge.green { background: #dcfce7; color: #15803d; }
.hm-bc-badge.blue  { background: #dbeafe; color: #0369a1; }
.hm-bc-badge.amber { background: #fef9c3; color: #92400e; }

.hm-bc-icon {
    width: 54px; height: 54px; border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem; margin-bottom: 18px;
}
.hm-bento-card.bc-green .hm-bc-icon { background: linear-gradient(135deg, #dcfce7, #bbf7d0); box-shadow: 0 4px 14px rgba(22,163,74,0.22); }
.hm-bento-card.bc-blue  .hm-bc-icon { background: linear-gradient(135deg, #dbeafe, #bae6fd); box-shadow: 0 4px 14px rgba(14,165,233,0.22); }
.hm-bento-card.bc-amber .hm-bc-icon { background: linear-gradient(135deg, #fef9c3, #fde68a); box-shadow: 0 4px 14px rgba(245,158,11,0.22); }

.hm-bc-title { font-weight: 800; font-size: 1.15rem; color: #0f172a; margin-bottom: 10px; }
.hm-bc-desc  { color: #64748b; font-size: 0.87rem; line-height: 1.68; margin-bottom: 18px; }
.hm-bc-tags  { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 22px; }
.hm-bc-tag   { font-size: 0.69rem; font-weight: 700; padding: 4px 11px; border-radius: 100px; }
.hm-bento-card.bc-green .hm-bc-tag { background: #dcfce7; color: #15803d; }
.hm-bento-card.bc-blue  .hm-bc-tag { background: #dbeafe; color: #0369a1; }
.hm-bento-card.bc-amber .hm-bc-tag { background: #fef9c3; color: #92400e; }

/* Scan animation for disease card */
.hm-scan-box {
    background: #f8fafc; border: 1.5px solid #e2e8f0;
    border-radius: 16px; padding: 16px; overflow: hidden;
    position: relative; height: 110px; margin-top: 4px;
    display: flex; align-items: center; justify-content: center;
}
.hm-scan-emoji { font-size: 3rem; opacity: 0.45; }
.hm-scan-beam {
    position: absolute; left: 8px; right: 8px; height: 2px;
    background: linear-gradient(90deg, transparent, #22c55e 40%, #22c55e 60%, transparent);
    box-shadow: 0 0 10px rgba(34,197,94,0.6);
    animation: scanBeam 2.8s ease-in-out infinite;
}
@keyframes scanBeam {
    0%   { top: 15%; opacity: 0; }
    8%   { opacity: 1; }
    92%  { opacity: 1; }
    100% { top: 85%; opacity: 0; }
}
.hm-scan-pin {
    position: absolute; width: 9px; height: 9px; border-radius: 50%;
    background: #22c55e; box-shadow: 0 0 8px rgba(34,197,94,0.9);
    animation: pinPop 2.8s ease-in-out infinite;
}
.hm-scan-pin:nth-child(1) { animation-delay: 0.6s; }
.hm-scan-pin:nth-child(2) { top: 55%; left: 70%; animation-delay: 1.1s; }
.hm-scan-pin:nth-child(3) { top: 30%; left: 80%; animation-delay: 1.6s; }
@keyframes pinPop {
    0%, 40%, 100% { transform: scale(0); opacity: 0; }
    55%, 85%      { transform: scale(1); opacity: 1; }
}
.hm-scan-result-tag {
    position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%);
    background: white; border: 1.5px solid #fca5a5;
    border-radius: 8px; padding: 4px 12px;
    font-size: 0.72rem; font-weight: 700; color: #dc2626;
    white-space: nowrap; box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    animation: resultSlide 2.8s ease-in-out infinite;
}
@keyframes resultSlide {
    0%, 55%  { transform: translateX(-50%) translateY(40px); opacity: 0; }
    70%, 88% { transform: translateX(-50%) translateY(0); opacity: 1; }
    100%     { transform: translateX(-50%) translateY(40px); opacity: 0; }
}

/* Live price table */
.hm-price-list { display: flex; flex-direction: column; gap: 9px; }
.hm-price-row {
    display: flex; align-items: center; gap: 10px;
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 9px 14px;
    transition: all 0.3s; animation: priceHighlight 4s ease-in-out infinite;
}
.hm-price-row:nth-child(2) { animation-delay: 1.3s; }
.hm-price-row:nth-child(3) { animation-delay: 2.6s; }
@keyframes priceHighlight {
    0%, 88%, 100% { background: #f8fafc; border-color: #e2e8f0; }
    92%, 96%      { background: #f0fdf4; border-color: #bbf7d0; }
}
.hm-pcity { flex: 1; font-size: 0.82rem; font-weight: 500; color: #475569; }
.hm-pval  { font-weight: 800; font-size: 0.9rem; color: #15803d; }
.hm-pchg  { font-size: 0.68rem; font-weight: 700; padding: 2px 8px; border-radius: 100px; }
.hm-pchg.up   { background: #dcfce7; color: #16a34a; }
.hm-pchg.flat { background: #f1f5f9; color: #64748b; }

/* Language chat chips */
.hm-lang-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px; }
.hm-lchip {
    font-size: 0.84rem; font-weight: 700; padding: 7px 16px;
    border-radius: 100px; border: 1.5px solid #e2e8f0;
    color: #64748b; background: white; transition: all 0.22s;
}
.hm-lchip.active {
    background: linear-gradient(135deg, #dcfce7, #bbf7d0);
    border-color: #86efac; color: #15803d;
    box-shadow: 0 3px 10px rgba(22,163,74,0.2);
}
.hm-lchip-extra {
    font-size: 0.76rem; font-weight: 600; padding: 7px 14px;
    border-radius: 100px; background: #fef9c3;
    border: 1.5px solid #fde68a; color: #92400e;
}

/* Wave bar animation */
.hm-wave { display: flex; align-items: flex-end; gap: 4px; height: 36px; margin-top: 14px; }
.hm-wbar {
    width: 5px; border-radius: 3px; flex-shrink: 0;
    background: linear-gradient(to top, #16a34a, #86efac);
    animation: waveBar 1.4s ease-in-out infinite;
}
.hm-wbar:nth-child(1) { animation-delay: 0s;    }
.hm-wbar:nth-child(2) { animation-delay: 0.18s; }
.hm-wbar:nth-child(3) { animation-delay: 0.36s; }
.hm-wbar:nth-child(4) { animation-delay: 0.54s; }
.hm-wbar:nth-child(5) { animation-delay: 0.72s; }
.hm-wbar:nth-child(6) { animation-delay: 0.9s;  }
.hm-wbar:nth-child(7) { animation-delay: 1.08s; }
@keyframes waveBar {
    0%, 100% { height: 6px;  opacity: 0.5; }
    50%       { height: 32px; opacity: 1;   }
}

@keyframes heroGrad {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes advisorBg {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes pulseDot {
    0%, 100% { transform: scale(1); opacity: 1; box-shadow: 0 0 0 0 rgba(74,222,128,0.7); }
    50%       { transform: scale(1.3); opacity: 0.8; box-shadow: 0 0 0 6px rgba(74,222,128,0); }
}
@keyframes ticker {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}
@keyframes msgIn {
    0%   { opacity: 0; transform: translateY(10px); }
    100% { opacity: 1; transform: translateY(0); }
}
@keyframes btnGlow {
    0%, 100% { box-shadow: 0 6px 20px rgba(22,163,74,0.45); }
    50%       { box-shadow: 0 6px 32px rgba(22,163,74,0.7); }
}
@keyframes ctaGlow {
    0%, 100% { box-shadow: 0 8px 32px rgba(22,163,74,0.45); }
    50%       { box-shadow: 0 8px 52px rgba(22,163,74,0.72); }
}
@keyframes gradText {
    0%   { background-position: 0% 50%; }
    100% { background-position: 300% 50%; }
}
@keyframes floatCard {
    0%, 100% { transform: perspective(800px) rotateY(-4deg) rotateX(2deg) translateY(0); }
    50%       { transform: perspective(800px) rotateY(-4deg) rotateX(2deg) translateY(-14px); }
}
"""


# ═══════════════════════════════════════════════════════════════
#  HTML BLOCKS
# ═══════════════════════════════════════════════════════════════

HOME_HTML = """
<!-- ═══ HERO ═══ -->
<div class="nw-hero">
  <div class="nw-bento-bg">
    <div class="nw-cell"><img src="https://images.unsplash.com/photo-1625246333195-78d9c38ad449?w=900&auto=format&fit=crop" alt="Farm" /></div>
    <div class="nw-cell"><img src="https://images.unsplash.com/photo-1574943320219-553eb213f72d?w=900&auto=format&fit=crop" alt="Agriculture" /></div>
    <div class="nw-cell"><img src="https://images.unsplash.com/photo-1592982537447-7440770cbfc9?w=900&auto=format&fit=crop" alt="Crops" /></div>
    <div class="nw-cell"><img src="https://images.unsplash.com/photo-1560493676-04071c5f467b?w=900&auto=format&fit=crop" alt="Harvest" /></div>
  </div>
  <div class="nw-hero-content">
    <div class="nw-badge nw-badge-green">&#x2728; AI-Powered Agriculture &nbsp;&middot;&nbsp; &#x0915;&#x093F;&#x0938;&#x093E;&#x0928; &#x092E;&#x093F;&#x0924;&#x094D;&#x0930;</div>
    <h1 class="nw-h1">Grow Smarter,<br><span class="nw-gradient-text">Starts Here.</span></h1>
    <p class="nw-hero-desc">
      Your AI-powered farming companion &mdash; diagnose crop diseases instantly,
      check live mandi prices, and get expert advice in Hindi, Marathi, or English.
    </p>
    <div class="nw-hero-btns">
      <button class="nw-btn-primary" onclick="var b=document.getElementById('km-cta-hero');if(b)b.click();return false;">&#x1F33E; Get Started Free</button>
    </div>
  </div>
</div>

<!-- ═══ FEATURES ═══ -->
<section class="nw-features">
  <div class="nw-container">
    <div class="nw-section-head">
      <div class="nw-badge nw-badge-blue">&#x1F9E0; What Kisaan Mitra Does</div>
      <h2 class="nw-section-title">Smart Farming Solutions</h2>
      <p class="nw-section-desc">Everything an Indian farmer needs &mdash; disease diagnosis, live market prices, voice input in 3 languages, and 100% organic advice. All in one free AI app.</p>
    </div>
    <div class="nw-feat-grid">

      <div class="nw-act-card nw-ac-green">
        <div class="nw-ac-head">
          <div class="nw-ac-icon">&#x1F52C;</div>
          <span class="nw-ac-tag">Core Feature</span>
        </div>
        <h3 class="nw-ac-title">Crop Disease Diagnosis</h3>
        <p class="nw-ac-desc">Upload a photo or describe your crop's symptoms. Kisaan Mitra identifies the disease and prescribes organic remedies instantly &mdash; no agronomist visit needed.</p>
        <div class="nw-ac-chips">
          <span class="nw-ac-chip">&#x1F4F7; Photo AI</span>
          <span class="nw-ac-chip">&#x1F33F; Organic Rx</span>
          <span class="nw-ac-chip">&#x26A1; Instant</span>
        </div>
        <div class="nw-ac-footer">Try Disease Detection <span>&#x2192;</span></div>
      </div>

      <div class="nw-act-card nw-ac-blue">
        <div class="nw-ac-head">
          <div class="nw-ac-icon">&#x1F4CA;</div>
          <span class="nw-ac-tag">Live Data</span>
        </div>
        <h3 class="nw-ac-title">Live Mandi Price Advisor</h3>
        <p class="nw-ac-desc">Get real-time market rates for your crop and district. Combined with live weather data, the AI tells you whether to sell now, wait, or store.</p>
        <div class="nw-ac-chips">
          <span class="nw-ac-chip">&#x1F4CA; Live Rates</span>
          <span class="nw-ac-chip">&#x1F327;&#xFE0F; Weather</span>
          <span class="nw-ac-chip">&#x1F4CD; District-wise</span>
        </div>
        <div class="nw-ac-footer">Check Mandi Prices <span>&#x2192;</span></div>
      </div>

      <div class="nw-act-card nw-ac-yellow">
        <div class="nw-ac-head">
          <div class="nw-ac-icon">&#x1F3A4;</div>
          <span class="nw-ac-tag">Multilingual</span>
        </div>
        <h3 class="nw-ac-title">Voice Input in 3 Languages</h3>
        <p class="nw-ac-desc">Speak naturally in Hindi, Marathi, or English &mdash; no typing needed. The AI auto-detects your language and always replies in the same language you used.</p>
        <div class="nw-ac-chips">
          <span class="nw-ac-chip">&#x0939;&#x093F;&#x0902;&#x0926;&#x0940; Hindi</span>
          <span class="nw-ac-chip">&#x092E;&#x0930;&#x093E;&#x0920;&#x0940; Marathi</span>
          <span class="nw-ac-chip">&#x1F1EC;&#x1F1E7; English</span>
        </div>
        <div class="nw-ac-footer">Ask in Your Language <span>&#x2192;</span></div>
      </div>

      <div class="nw-act-card nw-ac-purple">
        <div class="nw-ac-head">
          <div class="nw-ac-icon">&#x1F4F7;</div>
          <span class="nw-ac-tag">Vision AI</span>
        </div>
        <h3 class="nw-ac-title">AI Photo Analysis</h3>
        <p class="nw-ac-desc">Point your camera at a diseased leaf or pest-affected crop. Computer vision examines the image and provides a detailed symptom report before prescribing treatment.</p>
        <div class="nw-ac-chips">
          <span class="nw-ac-chip">&#x1F9E0; Computer Vision</span>
          <span class="nw-ac-chip">&#x1F41B; Pest ID</span>
          <span class="nw-ac-chip">&#x1F4CB; Symptom Report</span>
        </div>
        <div class="nw-ac-footer">Upload a Crop Photo <span>&#x2192;</span></div>
      </div>

      <div class="nw-act-card nw-ac-red">
        <div class="nw-ac-head">
          <div class="nw-ac-icon">&#x1F33F;</div>
          <span class="nw-ac-tag">Always Organic</span>
        </div>
        <h3 class="nw-ac-title">100% Organic Solutions</h3>
        <p class="nw-ac-desc">Every remedy recommended is completely organic &mdash; neem oil, jeevamrut, panchagavya, cow urine spray. No synthetic chemicals, ever. Safe for soil and family.</p>
        <div class="nw-ac-chips">
          <span class="nw-ac-chip">&#x1F9AA; Neem Oil</span>
          <span class="nw-ac-chip">&#x1F404; Jeevamrut</span>
          <span class="nw-ac-chip">&#x274C; No Chemicals</span>
        </div>
        <div class="nw-ac-footer">See Organic Remedies <span>&#x2192;</span></div>
      </div>

      <div class="nw-act-card nw-ac-indigo">
        <div class="nw-ac-head">
          <div class="nw-ac-icon">&#x1F4AC;</div>
          <span class="nw-ac-tag">Smart AI</span>
        </div>
        <h3 class="nw-ac-title">Smart Multi-turn Chat</h3>
        <p class="nw-ac-desc">Kisaan Mitra remembers your full conversation. Ask follow-up questions, describe new symptoms, or switch topics seamlessly &mdash; the AI keeps full context throughout.</p>
        <div class="nw-ac-chips">
          <span class="nw-ac-chip">&#x1F9E0; Context Memory</span>
          <span class="nw-ac-chip">&#x1F501; Follow-ups</span>
          <span class="nw-ac-chip">&#x1F50A; Voice Reply</span>
        </div>
        <div class="nw-ac-footer">Start a Conversation <span>&#x2192;</span></div>
      </div>

    </div>
  </div>
</section>
<!-- ═══ CTA ═══ -->
<section class="nw-cta-section">
  <div class="nw-cta-card">
    <div class="nw-cta-icon">&#x1F331;</div>
    <h2 class="nw-cta-title">Ready to Transform Your Farm?</h2>
    <p class="nw-cta-desc">Join the agricultural revolution. Start using AI to grow smarter, save resources, and increase your profits today.</p>
    <button class="nw-btn-gradient" onclick="var b=document.getElementById('km-cta-bottom');if(b)b.click();return false;">Get Started Now &#x2192;</button>
    <p class="nw-cta-note">No credit card required &nbsp;&middot;&nbsp; 14-day free trial</p>
  </div>
</section>
"""


SIDEBAR_HEAD_HTML = """
<div class="km-sb-head">
  <div class="km-sb-logo">
    <div class="km-sb-logo-icon">&#x1F33E;</div>
    <div>
      <div class="km-sb-logo-text">Kisaan Mitra</div>
      <div class="km-sb-logo-sub">&#x0915;&#x093F;&#x0938;&#x093E;&#x0928; &#x092E;&#x093F;&#x0924;&#x094D;&#x0930; &middot; AI Farming Advisor</div>
    </div>
  </div>
  <div class="km-sb-status">
    <span class="km-sb-status-dot"></span> AI Online
  </div>
</div>
<div class="km-sb-caps">
  <span class="km-sb-cap">&#x1F52C; Disease</span>
  <span class="km-sb-cap">&#x1F4C8; Mandi Prices</span>
  <span class="km-sb-cap">&#x1F4F7; Photo AI</span>
  <span class="km-sb-cap">&#x1F3A4; Voice</span>
</div>
"""

CHAT_HEAD_HTML = """
<div class="km-chat-head">
  <div class="km-chat-head-icon">&#x1F4AC;</div>
  <div>
    <div class="km-chat-head-title">Conversation</div>
    <div class="km-chat-head-sub">Ask about diseases, mandi prices, or farming advice</div>
  </div>
  <div class="km-ai-badge">
    <span class="km-ai-dot"></span> AI Ready
  </div>
</div>
"""

FOOTER_HTML = """
<div class="km-footer">
  &#x1F33E; <strong>Kisaan Mitra</strong> &mdash; Powered by AI &middot; Helping Indian farmers grow smarter
</div>
"""


# ═══════════════════════════════════════════════════════════════
#  GRADIO APP
# ═══════════════════════════════════════════════════════════════

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.green,
        secondary_hue=gr.themes.colors.emerald,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "ui-sans-serif", "sans-serif"],
    ),
    css=CUSTOM_CSS,
    title="Kisaan Mitra \u2014 AI Farming Advisor",
) as demo:
    with gr.Tabs(selected=0, elem_classes="km-tabs") as tabs:
        # ── TAB 0: HOME ────────────────────────────────────────────────────────
        with gr.Tab(label="\U0001f3e0 Home", id=0):
            gr.HTML(HOME_HTML)

            with gr.Row(elem_id="km-home-cta-row"):
                hero_cta_btn = gr.Button(
                    "\U0001f33e Get Started Free",
                    elem_id="km-cta-hero",
                    variant="primary",
                )
                bottom_cta_btn = gr.Button(
                    "Get Started Now \u2192",
                    elem_id="km-cta-bottom",
                    variant="primary",
                )

        # ── TAB 1: ADVISOR ────────────────────────────────────────────────────
        with gr.Tab(label="\U0001f33e Advisor", id=1):
            with gr.Row(elem_id="km-split-row", equal_height=False):
                # ── LEFT: Sidebar ─────────────────────────────────────────────
                with gr.Column(scale=2, elem_id="km-sidebar", min_width=260):
                    gr.HTML(SIDEBAR_HEAD_HTML)

                    audio_input = gr.Audio(
                        label="\U0001f3a4 Voice Input",
                        type="filepath",
                        sources=["microphone"],
                        elem_id="km-audio-in",
                    )

                    image_input = gr.Image(
                        label="\U0001f4f7 Crop Photo",
                        type="filepath",
                        sources=["upload"],
                        height=150,
                        elem_id="km-image-in",
                    )

                    gr.HTML('<div class="km-sb-rule"></div>')

                    language_choice = gr.Dropdown(
                        label="\U0001f310 Language",
                        choices=["Hindi", "Marathi", "English"],
                        value="Hindi",
                        elem_id="km-lang",
                    )

                    audio_output = gr.Audio(
                        label="\U0001f50a Audio Reply",
                        type="filepath",
                        elem_id="km-audio-out",
                    )

                    gr.HTML('<div class="km-sb-rule"></div>')

                    clear_btn = gr.Button(
                        "\U0001f5d1\ufe0f Clear Chat",
                        elem_id="km-clear",
                    )

                # ── RIGHT: Chat Panel ─────────────────────────────────────────
                with gr.Column(scale=3, elem_id="km-chat-col", min_width=340):
                    gr.HTML(CHAT_HEAD_HTML)

                    # Quick-prompt chips (functional Gradio buttons)
                    gr.HTML(
                        '<div class="km-quick-bar"><span class="km-quick-label">Try:</span></div>'
                    )
                    with gr.Row(elem_id="km-quick-row"):
                        q1 = gr.Button(
                            "\U0001f52c Diagnose my crop",
                            elem_id="km-q1",
                        )
                        q2 = gr.Button(
                            "\U0001f4c8 Check mandi price",
                            elem_id="km-q2",
                        )
                        q3 = gr.Button(
                            "\U0001f33f Organic solutions",
                            elem_id="km-q3",
                        )

                    chatbot = gr.Chatbot(
                        label="",
                        height=480,
                        bubble_full_width=False,
                        show_copy_button=True,
                        elem_id="km-chatbot",
                        placeholder=(
                            "<div style='text-align:center;padding:48px 24px;font-family:Plus Jakarta Sans,sans-serif;'>"
                            "<div style='font-size:3.5rem;margin-bottom:16px;"
                            "filter:drop-shadow(0 6px 12px rgba(22,163,74,0.3));'>&#x1F33E;</div>"
                            "<p style='font-size:1.05rem;font-weight:800;color:#166534;"
                            "margin:0 0 6px;letter-spacing:-0.02em;'>Kisaan Mitra</p>"
                            "<p style='font-size:0.72rem;color:#86efac;margin:0 0 18px;font-weight:700;"
                            "letter-spacing:0.1em;text-transform:uppercase;'>"
                            "&#x0915;&#x093F;&#x0938;&#x093E;&#x0928; &#x092E;&#x093F;&#x0924;&#x094D;&#x0930; &nbsp;&middot;&nbsp; AI Powered</p>"
                            "<p style='font-size:0.84rem;color:#94a3b8;max-width:300px;"
                            "margin:0 auto;line-height:1.7;'>"
                            "Use the quick buttons above, type below, speak using the mic, "
                            "or upload a crop photo.</p>"
                            "</div>"
                        ),
                    )

                    with gr.Row(elem_id="km-input-row"):
                        text_input = gr.Textbox(
                            label="",
                            placeholder="Ask about a crop disease, mandi price, or organic solution...",
                            lines=2,
                            scale=5,
                            elem_id="km-textbox",
                        )
                        send_btn = gr.Button(
                            "Send \u27a4",
                            variant="primary",
                            scale=1,
                            min_width=114,
                            elem_id="km-send",
                        )

                    with gr.Accordion(
                        "\U0001f4da Example Queries \u2014 click to expand",
                        open=False,
                        elem_classes="km-accord",
                    ):
                        gr.Examples(
                            examples=[
                                ["Mere tamatar ke patte peele ho rahe hain", "Hindi"],
                                ["Should I sell my onions today in Mumbai?", "English"],
                                [
                                    "What organic spray works for aphids on wheat?",
                                    "English",
                                ],
                                ["Pune mein aaj pyaaz bechun ya nahi?", "Hindi"],
                                ["Nashik mein angur ka bhav kya hai aaj?", "Hindi"],
                                [
                                    "Majhya tambyachya zhadala kida laagla aahe",
                                    "Marathi",
                                ],
                            ],
                            inputs=[text_input, language_choice],
                            label=None,
                        )

                    gr.HTML(FOOTER_HTML)

    # ── State ─────────────────────────────────────────────────────────────────
    llm_history = gr.State([])
    cached_image = gr.State({"path": None, "analysis": ""})

    # ── Event wiring ──────────────────────────────────────────────────────────
    shared_inputs = [
        audio_input,
        image_input,
        text_input,
        language_choice,
        chatbot,
        llm_history,
        cached_image,
    ]
    shared_outputs = [
        chatbot,
        llm_history,
        cached_image,
        text_input,
        audio_output,
        language_choice,
    ]

    send_btn.click(fn=chat_fn, inputs=shared_inputs, outputs=shared_outputs)
    text_input.submit(fn=chat_fn, inputs=shared_inputs, outputs=shared_outputs)
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, llm_history, cached_image, text_input, audio_output],
    )
    # Both CTA buttons switch to the Advisor tab (id=1)
    hero_cta_btn.click(fn=lambda: gr.update(selected=1), outputs=tabs)
    bottom_cta_btn.click(fn=lambda: gr.update(selected=1), outputs=tabs)

    # Quick prompt chips fill the text box
    q1.click(
        fn=lambda: (
            "My crop has unusual spots or discoloration, can you help diagnose it?"
        ),
        outputs=text_input,
    )
    q2.click(
        fn=lambda: (
            "What are today's mandi prices for my crops and when is the best time to sell?"
        ),
        outputs=text_input,
    )
    q3.click(
        fn=lambda: (
            "What organic and natural solutions are available for my crop problem?"
        ),
        outputs=text_input,
    )


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("SERVER_PORT", 7860)),
        share=False,
    )
