---
title: Kisaan Mitra
emoji: 🌾
colorFrom: green
colorTo: emerald
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Kisaan Mitra (किसान मित्र) — AI Farming Advisor

A **RAG-based multi-turn conversational AI** for Indian farmers. Diagnose crop diseases via text/photo, check live mandi prices with weather-aware sell/wait/store advice, and get organic remedies — all in **Hindi, Marathi, or English** with voice input and spoken output.

---

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Language | Python | 3.11+ |
| UI Framework | [Gradio](https://gradio.app) | 4.44.0 |
| LLM Gateway | [OpenRouter](https://openrouter.ai) (free models: Llama 3.3 70B, Hermes 3 405B, Nemotron, Gemma, Qwen) | REST API |
| Vision Model | NVIDIA Nemotron Nano 12B VL (via OpenRouter) | — |
| Speech-to-Text | Google Web Speech API ([SpeechRecognition](https://pypi.org/project/SpeechRecognition/)) | 3.10.4 |
| Text-to-Speech | [gTTS](https://pypi.org/project/gTTS/) (Google TTS) | 2.5.1 |
| Weather Data | [OpenWeatherMap](https://openweathermap.org) API | — |
| Audio Processing | pydub | 0.25.1 |
| Image Processing | Pillow | 10.4.0 |
| Environment | python-dotenv | 1.0.1 |

### Architecture

```
Farmer Input (Voice / Text / Image)
    │
    ├─→ STT (SpeechRecognition) ─→ text
    │
    ▼
┌──────────────────────────┐
│       llm.py             │
│  • detect_language()     │  → hi / mr / en
│  • classify_intent()     │  → DISEASE / MARKET / UNKNOWN
└──────────────────────────┘
    │
    ├── DISEASE ─→ disease_module.py
    │               ├─ Vision analysis (photo → symptoms)
    │               ├─ KB lookup (disease_knowledge.json)
    │               └─ LLM → organic remedies
    │
    ├── MARKET  ─→ market_module.py
    │               ├─ Parse crop + district from query
    │               ├─ Fetch mandi prices (local JSON)
    │               ├─ Fetch live weather (OpenWeatherMap)
    │               └─ LLM → sell/wait/store recommendation
    │
    └── UNKNOWN ─→ Off-topic guardrail response
    │
    ▼
TTS (gTTS) ─→ Audio playback + Text response in Gradio UI
```

### Knowledge Bases

- **`data/disease_knowledge.json`** — 7 crop diseases (Leaf Spot, Late Blight, TYLCV, Powdery Mildew, Root Rot, Aphids, Nutrient Deficiency) with symptoms, organic remedies, prevention tips, and KVK referral thresholds.
- **`data/mandi_prices.json`** — Mandi prices for 6 Maharashtra districts (Pune, Nashik, Aurangabad, Nagpur, Mumbai, Kolhapur) across 7+ crops, with price trends and units.

---

## Prompt Design

All prompts are centralized in `prompts.py` and designed around strict guardrails and templated generation.

### System Guardrails (`SYSTEM_PROMPT_BASE`)

1. **Topic restriction** — Only farming/crop/pest/weather questions. Everything else gets a polite deflection.
2. **No chemicals** — Only organic/natural remedies (neem oil, jeevamrut, panchagavya, etc.). No synthetic pesticides or fertilizers.
3. **No medical advice** — Human/animal health questions are redirected to doctors/veterinarians.
4. **Uncertainty handling** — If unsure, explicitly say so and recommend the local Krishi Vigyan Kendra (KVK).
5. **Brevity** — 3–5 sentences unless detail is essential.
6. **Location-aware** — Tailor advice to the farmer's district/region climate and season.
7. **Organic-only framing** — Always frame around sustainable, natural practices.
8. **Language mirroring (critical)** — Reply in the **exact same language** the farmer used (Hindi, Marathi, or English). Never mix languages.

### Intent Classification Prompt

Classifies each query into exactly one of:
- **`DISEASE`** — crop disease, plant symptoms, pests, yellowing, wilting, spots, rot
- **`MARKET`** — crop prices, selling, mandi rates, market trends, weather impact on selling
- **`UNKNOWN`** — off-topic or unclear (triggers guardrail response)

### Disease Diagnosis Template (`DISEASE_PROMPT_TEMPLATE`)

```
Farmer's description: {disease_description}
{image_analysis_section}

1. Identify the most likely crop disease/pest/deficiency
2. Suggest 2–3 specific organic remedies with preparation/application instructions
3. If severe or unclear → recommend KVK visit
```

### Market Advisory Template (`MARKET_PROMPT_TEMPLATE`)

```
Crop: {crop}
District: {district}
Mandi Price Data: {price_data}
Weather Forecast: {weather_data}

1. SELL NOW / WAIT / STORE headline
2. 2–3 sentence explanation linking price trend + weather
3. Rain risk → transport/storage quality warning
4. Rising price + good weather → suggest waiting
5. Falling price or rain expected → suggest selling soon
```

### RAG Strategy

- **Context window**: First 2 messages + last 4 turns (`RAG_WINDOW = 4`) are retained for multi-turn coherence.
- **Disease KB retrieval**: Keyword-scored matching against disease names, aliases, crops, and symptoms from `disease_knowledge.json`.
- **Image caching**: Vision analysis results are cached per image hash to avoid repeated API calls for the same photo.

---

## Localization

Kisaan Mitra is designed for India's linguistically diverse farming community.

### Supported Languages

| Language | Script | Detection | STT | TTS |
|----------|--------|-----------|-----|-----|
| हिन्दी (Hindi) | Devanagari | Unicode + LLM detection | Google Web Speech (hi-IN) | gTTS (hi) |
| मराठी (Marathi) | Devanagari | Unicode + LLM detection | Google Web Speech (hi-IN fallback) | gTTS (mr) |
| English | Latin | Unicode + LLM detection | Google Web Speech (en-IN) | gTTS (en) |

### Language Detection Flow

1. **LLM-based**: `detect_language()` in `llm.py` calls OpenRouter with `LANGUAGE_DETECT_PROMPT` returning `hi`, `mr`, or `en`.
2. **Unicode fallback**: If the LLM call fails, script-based detection checks for Devanagari characters → Hindi/Marathi.
3. **Mirrored response**: The system prompt's Rule #8 enforces that every response is in the farmer's language.

### Speech Pipeline

```
Voice Input ──→ SpeechRecognition (Google Web Speech)
                  └─ Attempt hi-IN → fallback en-IN
                       │
                       ▼
              LLM processes text
                       │
                       ▼
              gTTS generates audio
                  └─ Auto-detect script:
                     Devanagari → hi/mr TTS
                     Latin      → en TTS
                       │
                       ▼
              Audio plays in Gradio UI
```

### Why This Matters

Over 60% of Indian farmers are more comfortable in regional languages than English. Voice interaction removes literacy barriers. Kisaan Mitra enables a farmer in rural Maharashtra to speak in Marathi, upload a photo of their crop, and receive spoken organic remedy advice — all without reading or typing.

---

## Getting Started

### Prerequisites

- Python 3.11+
- API keys (free tiers available):
  - [OpenRouter](https://openrouter.ai) — LLM & vision access
  - [OpenWeatherMap](https://openweathermap.org) — weather data

### Installation

```bash
# Clone the repository
git clone https://github.com/zainab-06-p/Kisaan-Mitra.git
cd Kisaan-Mitra

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the app
python app.py
```

The app will be available at `http://localhost:7870`.

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key for LLM + vision |
| `GEMINI_API_KEY` | No | Fallback Gemini API key |
| `OPENWEATHER_API_KEY` | Yes | OpenWeatherMap API key |

---

## Project Structure

```
├── app.py                  # Main Gradio UI + chat orchestration (2356 lines)
├── llm.py                  # OpenRouter LLM calls, vision analysis, language/intent detection
├── prompts.py              # System prompts, templates, guardrails
├── disease_module.py       # Disease diagnosis + knowledge base retrieval
├── market_module.py        # Mandi prices + weather advisory
├── stt.py                  # Speech-to-text via Google Web Speech
├── tts.py                  # Text-to-speech via gTTS
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed)
├── .gitignore
└── data/
    ├── disease_knowledge.json   # 7 crop diseases with organic remedies
    └── mandi_prices.json        # Mandi prices for 6 districts
```

---

## Deployment

The app is a long-running Gradio web service. Recommended platforms:

| Platform | Gradio Support | Custom Domain | Notes |
|----------|---------------|---------------|-------|
| [Hugging Face Spaces](https://huggingface.co/spaces) | ✅ Native | ✅ Pro tier | Free, built for Gradio |
| [Railway](https://railway.app) | ✅ Good | ✅ Yes | Paid, easy Docker deploy |
| [Render](https://render.com) | ✅ Good | ✅ Yes | Paid Web Service |
| [Google Cloud Run](https://cloud.run) | ✅ Good | ✅ Yes | Container-based |

> **Note**: Vercel is designed for serverless/static sites and does not support Gradio's long-running WebSocket connections. Use Hugging Face Spaces for the simplest free deployment with native Gradio support.

---

## License

MIT
