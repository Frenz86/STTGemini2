import streamlit as st
import speech_recognition as sr
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
from audio_recorder_streamlit import audio_recorder
import io
from typing import Dict, List, Any, Optional


load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API'))

# Modelli Gemini
volumio_model = genai.GenerativeModel("gemini-2.0-flash-exp")

# Inizializzazione session state
if "history" not in st.session_state:
    st.session_state.history = []

class AudioProcessor:
    """Elaborazione audio e riconoscimento vocale"""
    def process(self, audio_bytes: bytes) -> str:
        recognizer = sr.Recognizer()
        try:
            with io.BytesIO(audio_bytes) as audio_file:
                with sr.AudioFile(audio_file) as source:
                    audio = recognizer.record(source)
                text = recognizer.recognize_google(audio, language="it-IT")
                return text.lower()
        except Exception as e:
            st.error(f"Errore nel riconoscimento vocale: {str(e)}")
            return ""

class EmotionalAnalyzer:
    """Analisi emotiva con Gemini"""
    def analyze(self, text: str) -> Dict[str, Any]:
        try:
            prompt = """Analizza il testo utente e genera una raccomandazione musicale in formato JSON. 
            Usa SOLO una di queste categorie con le relative caratteristiche:

            Categorie consentite:
            1. Running: (120-140 BPM) Musica motivazionale per attività sportiva
            2. Kitchen: (80-100 BPM) Ritmi allegri per cucinare e socializzare
            3. Ambient: (60-80 BPM) Soundscape atmosferici e texture elettroniche
            4. Relaxing: (50-70 BPM) Melodie calmanti per meditazione e rilassamento
            5. Working: (90-110 BPM) Musica strumentale per concentrazione
            6. Walking: (100-120 BPM) Ritmi naturali e brani cantautorali

            Struttura JSON richiesta:
            {
                "flow_consigliato": "nome categoria",
                "bpm_range": "range BPM",
                "caratteristiche": ["caratteristica1", "caratteristica2"],
                "esempi_genere": ["genere1", "genere2"],
                "percezione_emotiva": "breve descrizione emozione rilevata (max 10 parole)",
                "reasoning": "spiegazione tecnica della scelta (max 20 parole)"
            }

            Esempio:
            Input: "Ho avuto una giornata stressante, vorrei qualcosa per rilassarmi"
            Output: {
                    "flow_consigliato": "Relaxing",
                    "bpm_range": "50-70 BPM",
                    "caratteristiche": ["melodie lente", "armonie minimali"],
                    "esempi_genere": ["ambient drone", "piano minimal"],
                    "percezione_emotiva": "stress, bisogno di rilassamento",
                    "reasoning": "BPM bassi e texture semplici riducono l'ansia"
                    }

            Analizza ora questo input: """
            
            # Creazione del placeholder per lo streaming
            response_placeholder = st.empty()
            full_response = ""
            
            # Generazione della risposta in streaming
            for response in volumio_model.generate_content(prompt + text, stream=True):
                chunk = response.text
                full_response += chunk
                # Aggiorna il placeholder con la risposta parziale
                response_placeholder.text(full_response)
            
            # Pulizia e parsing della risposta finale
            cleaned_response = full_response.replace('```json', '').replace('```', '').strip()
            return json.loads(cleaned_response)
            
        except Exception as e:
            st.error(f"Errore nell'analisi: {str(e)}")
            return {}

def volumio_response(user_input: str) -> str:
    """Risposte brevi per comandi Volumio"""
    # Creazione del placeholder per lo streaming
    response_placeholder = st.empty()
    full_response = ""
    
    # Generazione della risposta in streaming
    for response in volumio_model.generate_content(
        f"Risposta breve in italiano (max 2 righe) a: {user_input}",
        stream=True):

        chunk = response.text
        full_response += chunk
        # Aggiorna il placeholder con la risposta parziale
        response_placeholder.text(full_response)
    
    return full_response

def main():
    st.title("🎵 Volumio AI Assistant")
    # Registrazione audio
    audio_bytes = audio_recorder(pause_threshold=2.0)
    
    if audio_bytes:
        with st.spinner("Analisi in corso..."):
            processor = AudioProcessor()
            text = processor.process(audio_bytes)
            analysis = EmotionalAnalyzer().analyze(text)
            st.session_state.history.append({
                                            "type": "analisi",
                                            "input": text,
                                            "output": analysis
                                            })
            # Visualizzazione risultati
            st.subheader("Risultati Analisi")
            st.json(analysis)


if __name__ == "__main__":
    main()