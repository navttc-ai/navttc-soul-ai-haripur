import streamlit as st
import azure.cognitiveservices.speech as speechsdk
import os
import time

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="Azure Speech Translator (with Urdu)", layout="wide")

# Load secrets
try:
    SPEECH_KEY = st.secrets["SPEECH_KEY"]
    SPEECH_REGION = st.secrets["SPEECH_REGION"]
except FileNotFoundError:
    st.error("Secrets not found. Please set up .streamlit/secrets.toml")
    st.stop()

# Supported Languages (Name: Code)
# Added Urdu (ur-PK) to this list
LANGUAGES = {
    "English (US)": "en-US",
    "Urdu (Pakistan)": "ur-PK",
    "Spanish": "es-ES",
    "French": "fr-FR",
    "German": "de-DE",
    "Italian": "it-IT",
    "Chinese (Mandarin)": "zh-CN",
    "Japanese": "ja-JP",
    "Korean": "ko-KR",
    "Portuguese (Brazil)": "pt-BR",
    "Russian": "ru-RU",
    "Hindi": "hi-IN",
    "Arabic (Egypt)": "ar-EG"
}

# Initialize Session State for History
if "history" not in st.session_state:
    st.session_state.history = []

# --- 2. AZURE SPEECH FUNCTIONS ---

def recognize_and_translate(source_lang_code, target_lang_code):
    """
    Records audio, translates it, and synthesizes the result to a file.
    """
    # A. Configure Translation
    translation_config = speechsdk.translation.SpeechTranslationConfig(
        subscription=SPEECH_KEY, 
        region=SPEECH_REGION
    )
    
    # Set input language and target language
    translation_config.speech_recognition_language = source_lang_code
    translation_config.add_target_language(target_lang_code)

    # Audio Config (Default Microphone)
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

    # Create the Translation Recognizer
    recognizer = speechsdk.translation.TranslationRecognizer(
        translation_config=translation_config, 
        audio_config=audio_config
    )
    
    status_text = st.empty()
    status_text.info("🔴 Listening... Speak now!")

    # B. Recognize Once (Single Utterance)
    result = recognizer.recognize_once()

    # C. Process Result
    if result.reason == speechsdk.ResultReason.TranslatedSpeech:
        status_text.success("Processing complete!")
        
        original_text = result.text
        translated_text = result.translations[target_lang_code]
        
        # D. Synthesize Translation (Text-to-Speech)
        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
        
        # Get the correct voice (including Urdu)
        speech_config.speech_synthesis_voice_name = get_voice_for_language(target_lang_code)
        
        # Save to file for Streamlit to play
        file_name = "translation_output.wav"
        audio_output = speechsdk.audio.AudioOutputConfig(filename=file_name)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output)
        
        synthesizer.speak_text_async(translated_text).get()
        
        return original_text, translated_text, file_name
        
    elif result.reason == speechsdk.ResultReason.NoMatch:
        status_text.warning("No speech could be recognized.")
        return None, None, None
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        status_text.error(f"Error: {cancellation_details.reason}")
        return None, None, None
    
    return None, None, None

def get_voice_for_language(lang_code):
    """
    Returns the specific Neural Voice name for the selected language.
    Added ur-PK-UzmaNeural for Urdu.
    """
    voices = {
        "en-US": "en-US-AvaMultilingualNeural",
        "ur-PK": "ur-PK-UzmaNeural",  # Urdu Voice
        "es-ES": "es-ES-ElviraNeural",
        "fr-FR": "fr-FR-DeniseNeural",
        "de-DE": "de-DE-KatjaNeural",
        "it-IT": "it-IT-ElsaNeural",
        "zh-CN": "zh-CN-XiaoxiaoNeural",
        "ja-JP": "ja-JP-NanamiNeural",
        "ko-KR": "ko-KR-SunHiNeural",
        "pt-BR": "pt-BR-FranciscaNeural",
        "ru-RU": "ru-RU-SvetlanaNeural",
        "hi-IN": "hi-IN-SwaraNeural",
        "ar-EG": "ar-EG-SalmaNeural"
    }
    # Default to English if not found
    return voices.get(lang_code, "en-US-AvaMultilingualNeural")

# --- 3. UI LAYOUT ---

st.title("🎙️ Azure Real-Time Speech Translator")
st.markdown("Speak in one language, hear it in another.")

# Selectors
col1, col2 = st.columns(2)
with col1:
    # Set default index to 1 (Urdu) or 0 (English) as desired
    source_lang_name = st.selectbox("I speak:", list(LANGUAGES.keys()), index=0)
    source_lang_code = LANGUAGES[source_lang_name]

with col2:
    # Default translation target
    target_lang_name = st.selectbox("Translate to:", list(LANGUAGES.keys()), index=1)
    target_lang_code = LANGUAGES[target_lang_name]

st.divider()

# Big Red Button
b_col1, b_col2, b_col3 = st.columns([1, 2, 1])
with b_col2:
    start_btn = st.button("🔴 Start Recording (Speak One Sentence)", use_container_width=True, type="primary")

# Logic
if start_btn:
    original, translated, audio_file = recognize_and_translate(source_lang_code, target_lang_code)
    
    if original and translated:
        # Save to history
        st.session_state.history.insert(0, {
            "source": original,
            "target": translated,
            "s_lang": source_lang_name,
            "t_lang": target_lang_name
        })
        
        # Display Current Result
        st.subheader("Result:")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.info(f"**Original ({source_lang_name}):**\n\n{original}")
        with res_col2:
            st.success(f"**Translation ({target_lang_name}):**\n\n{translated}")
        
        # Play Audio
        st.audio(audio_file)

# --- 4. SESSION HISTORY ---
if st.session_state.history:
    st.divider()
    st.subheader("📜 Session History")
    
    for item in st.session_state.history:
        with st.expander(f"{item['s_lang']} ➡️ {item['t_lang']}: {item['source'][:30]}..."):
            h_col1, h_col2 = st.columns(2)
            with h_col1:
                st.caption("Original")
                st.write(item['source'])
            with h_col2:
                st.caption("Translated")
                st.write(item['target'])