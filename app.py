from dataclasses import dataclass
import streamlit as st
import speech_recognition as sr
import google.generativeai as genai
import os
import json
from audio_recorder_streamlit import audio_recorder
import io
from typing import Dict, List, Any, Optional, Tuple
import time
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables and configure Gemini
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
genai.configure(api_key=GEMINI_API_KEY)

VOLUMIO_MODEL = genai.GenerativeModel("gemini-2.0-flash-exp")

@dataclass
class MusicCategory:
    """Data class for music categories and their properties"""
    name: str
    bpm_range: str
    description: str

MUSIC_CATEGORIES = {
                    "Running": MusicCategory("Running", "120-140", "Musica motivazionale per attività sportiva"),
                    "Kitchen": MusicCategory("Kitchen", "80-100", "Ritmi allegri per cucinare e socializzare"),
                    "Ambient": MusicCategory("Ambient", "60-80", "Soundscape atmosferici e texture elettroniche"),
                    "Relaxing": MusicCategory("Relaxing", "50-70", "Melodie calmanti per meditazione e rilassamento"),
                    "Working": MusicCategory("Working", "90-110", "Musica strumentale per concentrazione"),
                    "Walking": MusicCategory("Walking", "100-120", "Ritmi naturali e brani cantautorali")
                    }

class AudioProcessor:
    """Enhanced audio processing and voice recognition"""
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Adjust recognition parameters for better accuracy
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8

    @staticmethod
    def _prepare_audio_file(audio_bytes: bytes) -> io.BytesIO:
        """Prepare audio file for processing"""
        audio_file = io.BytesIO(audio_bytes)
        audio_file.seek(0)  # Reset file pointer
        return audio_file

    def process(self, audio_bytes: bytes) -> Tuple[str, bool]:
        """Process audio and return transcribed text with success status"""
        try:
            audio_file = self._prepare_audio_file(audio_bytes)
            with sr.AudioFile(audio_file) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio, language="it-IT")
                return text.lower(), True
        except sr.UnknownValueError:
            logger.warning("Speech recognition could not understand audio")
            return "Audio non chiaro, per favore riprova.", False
        except sr.RequestError as e:
            logger.error(f"Could not request results from speech recognition service: {e}")
            return "Errore nel servizio di riconoscimento vocale.", False
        except Exception as e:
            logger.error(f"Unexpected error in audio processing: {e}")
            return f"Errore imprevisto: {str(e)}", False

class EmotionalAnalyzer:
    """Enhanced emotional analysis with Gemini"""
    
    @staticmethod
    def _create_analysis_prompt(text: str) -> str:
        """Create the analysis prompt with optimized formatting"""
        return f"""Analizza il testo utente e genera una raccomandazione musicale in formato JSON. 
        Usa SOLO una di queste categorie con le relative caratteristiche:

        {EmotionalAnalyzer._format_categories()}

        Struttura JSON richiesta:
        {{
            "flow_consigliato": "nome categoria",
            "bpm_range": "range BPM",
            "caratteristiche": ["caratteristica1", "caratteristica2"],
            "esempi_genere": ["genere1", "genere2"],
            "percezione_emotiva": "breve descrizione emozione rilevata (max 10 parole)",
            "reasoning": "spiegazione tecnica della scelta (max 20 parole)"
        }}

        Analizza ora questo input: {text}"""

    @staticmethod
    def _format_categories() -> str:
        """Format music categories for the prompt"""
        return "\n".join(
            f"{cat.name}: ({cat.bpm_range} BPM) {cat.description}"
            for cat in MUSIC_CATEGORIES.values()
        )

    @staticmethod
    def _clean_response(response: str) -> str:
        """Clean the API response"""
        return response.replace('```json', '').replace('```', '').strip()

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text and return music recommendations"""
        start_time = time.time()
        try:
            prompt = self._create_analysis_prompt(text)
            response_placeholder = st.empty()
            
            # Non-async response generation
            response = VOLUMIO_MODEL.generate_content(prompt)
            if response and response.text:
                cleaned_response = self._clean_response(response.text)
                try:
                    result = json.loads(cleaned_response)
                    # Aggiungi la latenza in millisecondi
                    result['latenza_ms'] = round((time.time() - start_time) * 1000)
                    return result
                except json.JSONDecodeError:
                    logger.error("Invalid JSON in response")
                    return self._create_error_response("Formato risposta non valido", start_time)
            else:
                logger.error("Empty response from model")
                return self._create_error_response("Nessuna risposta dal modello", start_time)
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return self._create_error_response(str(e), start_time)

    @staticmethod
    def _create_error_response(error_message: str, start_time: float) -> Dict[str, Any]:
        """Create a standardized error response with latency"""
        return {
            "flow_consigliato": "Relaxing",  # Default fallback
            "bpm_range": "60-80 BPM",
            "caratteristiche": ["fallback_mode"],
            "esempi_genere": ["ambient"],
            "percezione_emotiva": f"Errore: {error_message}",
            "reasoning": "Risposta di fallback a causa di un errore",
            "latenza_ms": round((time.time() - start_time) * 1000)
        }

@st.cache_data(ttl=3600)
def get_volumio_response(user_input: str) -> str:
    """Cached Volumio responses for common inputs"""
    try:
        response = VOLUMIO_MODEL.generate_content(
            f"Risposta breve in italiano (max 2 righe) a: {user_input}")
        
        if response and response.text:
            return response.text
        else:
            logger.error("Empty response from model")
            return "Mi dispiace, non ho capito. Puoi ripetere?"
            
    except Exception as e:
        logger.error(f"Error in Volumio response: {e}")
        return "Mi dispiace, si è verificato un errore nella generazione della risposta."

class VolumioDashboard:
    """Dashboard UI management"""
    def __init__(self):
        if "history" not in st.session_state:
            st.session_state.history = []
        
        self.audio_processor = AudioProcessor()
        self.emotional_analyzer = EmotionalAnalyzer()

    def render(self):
        """Render the dashboard"""
        st.title("🎵 Volumio AI Assistant")
        self._render_audio_recorder()
        self._render_history()

    def _render_audio_recorder(self):
        """Render audio recording section"""
        audio_bytes = audio_recorder(pause_threshold=2.0)
        
        if audio_bytes:
            with st.spinner("Analisi in corso..."):
                self._process_audio(audio_bytes)

    def _process_audio(self, audio_bytes: bytes):
        """Process recorded audio"""
        text, success = self.audio_processor.process(audio_bytes)
        
        if success:
            analysis = self.emotional_analyzer.analyze(text)
            st.session_state.history.append({
                "type": "analisi",
                "input": text,
                "output": analysis
            })
            
            self._display_analysis(analysis)
        else:
            st.error(text)  # Display error message

    def _display_analysis(self, analysis: Dict[str, Any]):
        """Display analysis results"""
        st.subheader("Risultati Analisi")
        st.json(analysis)

    def _render_history(self):
        """Render interaction history"""
        if st.session_state.history:
            st.subheader("Cronologia Interazioni")
            for item in st.session_state.history[-5:]:  # Show last 5 interactions
                st.text(f"Input: {item['input']}")
                st.json(item['output'])

def main():
    """Main application entry point"""
    try:
        dashboard = VolumioDashboard()
        dashboard.render()
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("Si è verificato un errore nell'applicazione. Per favore, ricarica la pagina.")

if __name__ == "__main__":
    main()