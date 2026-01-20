import os
import streamlit as st
import streamlit.components.v1 as components
import json
# import onnx_asr
from scipy.io import wavfile
import numpy as np
import tempfile
import base64
import time
import soundfile as sf
from pedalboard import Pedalboard, Resample
import sys
# import pysqlite3
from datetime import datetime
from dotenv import load_dotenv
from langdetect import detect
from deep_translator import GoogleTranslator
# from indic_numtowords import num2words

# ── Translation Cache System ──────────────────────────────────────────────

@st.cache_data
def get_ui_translations():
    """Cache all UI translations to reduce latency"""
    translations = {}
    
    try:
        translator = GoogleTranslator(source='en', target='kn')
        
        # Main UI elements
        translations.update({
            'interactive_simulation_educational_agent': translator.translate("Interactive Simulation Educational Agent"),
            'welcome_msg_template': "Welcome! Ready to learn about **{concept}**? Click 'Start Learning' to begin your personalized learning session.",
            'start_learning': translator.translate("Start Learning"),
            'session_info': translator.translate("Session Info"),
            'how_to_interact': translator.translate("How to interact:"),
            'type_responses': translator.translate("Type your responses in the chat input"),
            'use_microphone': translator.translate("Or use the microphone to speak"),
            'agent_guide': translator.translate("The agent will guide you through learning"),
            
            # Audio/Input elements
            'voice_input': translator.translate("Voice Input"),
            'click_to_speak': translator.translate("Click to speak"),
            'transcribing': translator.translate("Transcribing..."),
            'you_said': translator.translate("You said:"),
            'type_response_here': translator.translate("Type your response here..."),
            
            # Simulation elements
            'phase_before_change': translator.translate("Phase: Before Change"),
            'phase_changing_parameters': translator.translate("Phase: Changing Parameters..."),
            'phase_after_change': translator.translate("Phase: After Change"),
            'simulation_running_msg': translator.translate("**Simulation running above** - Watch the pendulum carefully and notice what changes!"),
            
            # Image context
            'why_image_helps': translator.translate("Why this image helps your learning"),
            
            # Session end elements
            'learning_session_complete': "Learning Session Complete!",
            'session_summary': "Session Summary",
            'download_session_summary': "Download Session Summary",
            'session_completed': "Session completed successfully!",
            'langfuse_session_details': "Langfuse Session Details",
            'computing_session_metrics': "Computing session metrics...",
            'session_metrics_computed': "Session metrics computed and uploaded to Langfuse!",
            'failed_compute_metrics': "Failed to compute metrics:",
            'session_metrics': "Session Metrics",
            'quiz_score': "Quiz Score",
            'user_type': "User Type",
            'engagement_rating': "Engagement Rating",
            'interest_rating': "Interest Rating",
            'concepts_covered': "Concepts Covered",
            'enjoyment_probability': "Enjoyment Probability",
            'download_session_metrics': "Download Session Metrics",
            'start_new_session': "Start New Session",

            # Footer
            'powered_by': "Powered by Educational AI Agent",
            'tracked_with': "Tracked with Langsmith",

            # Thinking/Processing
            'thinking': translator.translate("Thinking..."),
            
            # Physics simulation parameters
            'length': translator.translate("Length"),
            'gravity': translator.translate("Gravity"),
            'amplitude': translator.translate("Amplitude"),
            'mass': translator.translate("Mass"),
            'period': translator.translate("Period")
        })
        
    except Exception as e:
        st.warning(f"Translation service unavailable, using English fallback: {e}")
        # English fallbacks
        translations.update({
            'interactive_simulation_educational_agent': "Interactive Simulation Educational Agent",
            'welcome_msg_template': "Welcome! Ready to learn about **{concept}**? Click 'Start Learning' to begin your personalized learning session.",
            'start_learning': "Start Learning",
            'session_info': "Session Info",
            'how_to_interact': "How to interact:",
            'type_responses': "Type your responses in the chat input",
            'use_microphone': "Or use the microphone to speak",
            'agent_guide': "The agent will guide you through learning",
            'voice_input': "Voice Input",
            'click_to_speak': "Click to speak",
            'transcribing': "Transcribing...",
            'you_said': "You said:",
            'type_response_here': "Type your response here...",
            'phase_before_change': "Phase: Before Change",
            'phase_changing_parameters': "Phase: Changing Parameters...",
            'phase_after_change': "Phase: After Change",
            'simulation_running_msg': "**Simulation running above** - Watch the pendulum carefully and notice what changes!",
            'why_image_helps': "Why this image helps your learning",
            'learning_session_complete': "Learning Session Complete!",
            'session_summary': "Session Summary",
            'download_session_summary': "Download Session Summary",
            'session_completed': "Session completed successfully!",
            'langfuse_session_details': "Langfuse Session Details",
            'computing_session_metrics': "Computing session metrics...",
            'session_metrics_computed': "Session metrics computed and uploaded to Langfuse!",
            'failed_compute_metrics': "Failed to compute metrics:",
            'session_metrics': "Session Metrics",
            'quiz_score': "Quiz Score",
            'user_type': "User Type",
            'engagement_rating': "Engagement Rating",
            'interest_rating': "Interest Rating",
            'concepts_covered': "Concepts Covered",
            'enjoyment_probability': "Enjoyment Probability",
            'download_session_metrics': "Download Session Metrics",
            'start_new_session': "Start New Session",
            'powered_by': "Powered by Educational AI Agent",
            'tracked_with': "Tracked with Langfuse",
            'thinking': "Thinking...",
            
            # Physics simulation parameters (English fallbacks)
            'length': "Length",
            'gravity': "Gravity", 
            'amplitude': "Amplitude",
            'mass': "Mass",
            'period': "Period"
        })
    
    return translations

# Initialize translations at startup
UI_TRANSLATIONS = get_ui_translations()


# Import the audio_recorder component
from audio_recorder_streamlit import audio_recorder

# Import gTTS for text-to-speech
from gtts import gTTS

# sys.modules["sqlite3"] = pysqlite3

import hashlib

from decimal import Decimal

# Map Western digits -> Kannada digits
_DIGIT_MAP = str.maketrans({
    "0": "೦", "1": "೧", "2": "೨", "3": "೩", "4": "೪",
    "5": "೫", "6": "೬", "7": "೭", "8": "೮", "9": "೯",
})

def translate_number_to_kannada(x, *,
                              float_format="0.1f",
                              strip_trailing_zeros=False,
                              keep_commas=True):
    """
    Convert a number or a string containing digits to Kannada digits.

    Parameters
    ----------
    x : int | float | Decimal | str
        The value to convert. If a string, only digits are transformed;
        other characters (., -, spaces, letters) are preserved.
    float_format : str | None
        If x is float/Decimal and you want custom formatting, pass a format
        specifier (e.g. '.2f', 'f', 'g'). If None, uses str(x).
    strip_trailing_zeros : bool
        If True and using fixed-point formatting, trims trailing zeros and a
        trailing decimal point (e.g. '12.3400' -> '12.34', '10.000' -> '10').
    keep_commas : bool
        If False, commas are removed from the string before translation.

    Returns
    -------
    str : the Kannada-digit string.
    """
    # 1) Turn input into a string with your desired formatting
    if isinstance(x, (float, Decimal)):
        s = format(x, float_format) if float_format else str(x)
    elif isinstance(x, int):
        s = str(x)
    else:
        s = str(x)

    # 2) Optional cleanup
    if not keep_commas:
        s = s.replace(",", "")

    if strip_trailing_zeros and "." in s:
        # Only safe to strip if it's plain decimal notation
        # (avoid touching scientific notation)
        before_e, sep, after_e = s.partition("e") if "e" in s or "E" in s else (s, "", "")
        # Work on 'before_e' only
        if "." in before_e:
            before_e = before_e.rstrip("0").rstrip(".")
        s = before_e + (sep + after_e if sep else "")

    # 3) Translate digits
    print(s.translate(_DIGIT_MAP))
    return s.translate(_DIGIT_MAP)

def msg_id_from_text(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def get_image_base64(image_path):
    """Convert image to base64 for embedding in HTML"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        st.error(f"Error loading image {image_path}: {e}")
        return None

if st.button('Clear Resource Cache'):
    st.cache_resource.clear()
    st.success("Resource cache cleared!")

import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

import whisper
import torch
from transformers import pipeline

class WhisperASR:
    def __init__(self, model_name: str = "vasista22/whisper-kannada-base"):
        # Use the fine-tuned Kannada Whisper model from Hugging Face
        self.model_name = model_name
        print(f"Loading Whisper model: {model_name}")
        try:
            # Create the ASR pipeline as per the official usage example
            self.transcribe = pipeline(
                task="automatic-speech-recognition",
                model=model_name,
                token=False
            )
            
            # Set the forced decoder IDs for Kannada transcription
            self.transcribe.model.config.forced_decoder_ids = self.transcribe.tokenizer.get_decoder_prompt_ids(
                language="kn", 
                task="transcribe"
            )
                        
        except Exception as e:
            print(f"Error loading fine-tuned model: {e}")
            st.error(e)
            print("Falling back to OpenAI Whisper tiny model...")
            # Fallback to OpenAI Whisper if HF model fails
            self.transcribe = None
            self.fallback_model = whisper.load_model("tiny")

    def recognize(self, audio_path: str) -> str:
        try:
            if self.transcribe is not None:
                # Use Hugging Face Transformers pipeline (preferred method)
                result = self.transcribe(audio_path)
                return result["text"].strip() if result and "text" in result else ""
            else:
                # Fallback to OpenAI Whisper
                result = self.fallback_model.transcribe(audio_path, language='kn', fp16=False)
                return result.get("text", "").strip()
                
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            st.error(e)
            return "[Speech recognition failed]"
        
# Load environment variables
load_dotenv(dotenv_path=".env", override=True)

# Import the EducationalAgent class
try:
    from educational_agent_optimized_langsmith_kannada_v4.agent import EducationalAgent
    from educational_agent_optimized_langsmith_kannada_v4.config import concept_pkg
    from tester_agent.session_metrics import compute_and_upload_session_metrics
except ImportError as e:
    st.error(f"Could not import EducationalAgent: {e}")
    st.stop()
    
# ── ASR & TTS Functions ──────────────────────────────────────────────────
@st.cache_resource(ttl=36000)
def load_asr_model():
    print("BOOT: about to init ASR...", flush=True)
    # Use the fine-tuned Kannada Whisper model
    model = WhisperASR(model_name="vasista22/whisper-kannada-tiny")
    print("BOOT: ASR ready", flush=True)
    return model

asr_model = load_asr_model()

def convert_to_mono_wav(input_path, output_path):
    try:
        sr, data = wavfile.read(input_path)
        if len(data.shape) == 2:
            data = np.mean(data, axis=1).astype(data.dtype)
        wavfile.write(output_path, sr, data)
    except Exception as e:
        st.error(f"Error converting WAV to mono: {e}")
        st.stop()

def transcribe_recorded_audio_bytes(audio_bytes):
    """Transcribe audio bytes to text using ASR model"""
    if asr_model is None:
        return "[Audio transcription disabled - ASR model not loaded]"
    
    tmp_path = f"temp_recorded_{time.time()}.wav"
    mono_wav_path = f"temp_mono_{time.time()}.wav"
    try:
        with open(tmp_path, 'wb') as f: 
            f.write(audio_bytes)
        convert_to_mono_wav(tmp_path, mono_wav_path)
        return asr_model.recognize(mono_wav_path)
    except Exception as e:
        st.error(f"Error in audio transcription: {e}")
        return "[Audio transcription failed]"
    finally:
        if os.path.exists(tmp_path): 
            os.remove(tmp_path)
        if os.path.exists(mono_wav_path): 
            os.remove(mono_wav_path)

from uuid import uuid4
import streamlit.components.v1 as components

def play_text_as_audio(text, container, message_id=None, speed_factor=1.25):
    """
    gTTS -> speedup with Pedalboard -> WAV playback.
    Also emits postMessage('audio_play'/'audio_end') to the sidebar iframe.
    """
    if not text or not text.strip():
        return

    try:
        # 1) TTS to mp3
        tts = gTTS(text=text, lang='kn', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            normal_speed_path = fp.name

        # 2) Read + resample (speed up)
        audio, sample_rate = sf.read(normal_speed_path)
        board = Pedalboard([Resample(target_sample_rate=int(sample_rate * speed_factor))])
        fast_audio = board(audio, sample_rate)

        # 3) Calculate correct duration for sped-up audio
        original_duration_ms = int((len(audio) / float(sample_rate)) * 1000)
        dur_ms = int(original_duration_ms / speed_factor)  # Actual playback duration after speedup
        new_sr = int(sample_rate * speed_factor)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            sf.write(fp.name, fast_audio, new_sr, format='WAV')
            fast_speed_path = fp.name

        # 4) Base64
        with open(fast_speed_path, "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")

        # 5) Clean up
        os.remove(normal_speed_path)
        os.remove(fast_speed_path)

        # 6) Render with JS hooks. Use components.html so <script> can run.
        msg_id = message_id or str(uuid4())
        audio_html = f"""
        <audio id="agentAudio" controls autoplay style="width: 100%; margin-top: 5px;">
          <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
        </audio>
        <script>
            (function() {{
            const MSG_ID = "{msg_id}";
            const a = document.getElementById('agentAudio');

            // Use BroadcastChannel so sibling iframes can talk directly
            const bc = new BroadcastChannel('agent_audio');

            // Throttle ticks to ~30fps
            let raf = null, lastTick = 0;
            const TICK_MS = 33;

            function send(type, extra = {{}}) {{
                bc.postMessage({{ type, id: MSG_ID, ...extra }});
            }}

            function tick(ts) {{
                if (!a.paused && !a.ended) {{
                if (!lastTick || (ts - lastTick) >= TICK_MS) {{
                    lastTick = ts;
                    send('audio_tick', {{
                    t: Math.round(a.currentTime * 1000),
                    dur: Math.round((a.duration || 0) * 1000)
                    }});
                }}
                raf = requestAnimationFrame(tick);
                }}
            }}

            a.addEventListener('loadedmetadata', () => {{
                // precise duration once known
                send('audio_meta', {{ dur: Math.round((a.duration || 0) * 1000) }});
            }});

            // IMPORTANT: start signal is "play"
            a.addEventListener('play', () => {{
                send('audio_play', {{
                t: Math.round(a.currentTime * 1000),
                dur: Math.round((a.duration || 0) * 1000)
                }});
                cancelAnimationFrame(raf);
                raf = requestAnimationFrame(tick);
            }});

            a.addEventListener('pause', () => {{
                send('audio_pause');
                cancelAnimationFrame(raf);
            }});

            a.addEventListener('ended', () => {{
                send('audio_end');
                cancelAnimationFrame(raf);
            }});
        }})();
            </script>

        """
        with container:
            components.html(audio_html, height=80)
        return msg_id

    except Exception as e:
        st.error(f"An error occurred in audio processing: {e}")
        return None

def create_pendulum_simulation_html(config):
    before_params = config['before_params']
    after_params = config['after_params']
    timing = config['timing']
    agent_message = config['agent_message']
    
    # Use cached translations for phase indicators
    phase_before = UI_TRANSLATIONS['phase_before_change']
    phase_changing = UI_TRANSLATIONS['phase_changing_parameters']
    phase_after = UI_TRANSLATIONS['phase_after_change']

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .simulation-container {{
                width: 100%;
                max-width: 600px;
                min-height: 600px;
                margin: 10px auto;
                background: #f0f6ff;
                border: 2px solid #c4afe9;
                border-radius: 15px;
                padding: 15px;
                text-align: center;
                position: relative;
            }}
            .simulation-canvas {{
                background: #ede9fe;
                border-radius: 12px;
                margin: 10px auto;
                display: block;
            }}
            .simulation-controls {{
                display: flex;
                justify-content: center;
                gap: 20px;
                margin: 10px 0;
                font-family: 'Segoe UI', sans-serif;
                flex-wrap: wrap;
            }}
            .param-display {{
                background: rgba(124, 58, 237, 0.1);
                padding: 8px 12px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 500;
            }}
            .agent-message {{
                background: rgba(124, 58, 237, 0.9);
                color: white;
                padding: 10px 15px;
                border-radius: 10px;
                margin: 10px 0;
                font-size: 16px;
                font-weight: 500;
            }}
            .phase-indicator {{
                background: #7c3aed;
                color: white;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 14px;
                font-weight: 600;
                margin: 10px 0;
            }}
            .hint-box {{
                text-align: left;
                background: #ffffff;
                border: 1px dashed #7c3aed;
                border-radius: 10px;
                padding: 10px 12px;
                margin-top: 8px;
                line-height: 1.5;
                color: #333;
            }}
            .hint-box b {{
                color: #7c3aed;
            }}
            .formula {{
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                background: #f7f2ff;
                padding: 2px 6px;
                border-radius: 6px;
                border: 1px solid #e4d7ff;
            }}
        </style>
    </head>
    <body>
        <div class="simulation-container">
            <div class="agent-message">{agent_message}</div>
            <div id="phase-indicator" class="phase-indicator">{phase_before}</div>
            
            <canvas id="pendulum-canvas" class="simulation-canvas" width="420" height="380"></canvas>
            
            <div class="simulation-controls">
                <div class="param-display">
                    {UI_TRANSLATIONS['length']}: <span id="length-display">{translate_number_to_kannada(f"{before_params['length']:.1f}")}m</span>
                </div>
                <div class="param-display">
                    {UI_TRANSLATIONS['gravity']}: <span id="gravity-display">{translate_number_to_kannada(f"{before_params['gravity']:.1f}")} m/s²</span>
                </div>
                <div class="param-display">
                    {UI_TRANSLATIONS['amplitude']}: <span id="amplitude-display">{translate_number_to_kannada(str(before_params['amplitude']))}°</span>
                </div>
                <div class="param-display">
                    {UI_TRANSLATIONS['mass']}: <span id="mass-display">{translate_number_to_kannada(f"{before_params.get('mass', 1):.2f}")} kg</span>
                </div>
                <div class="param-display">
                    {UI_TRANSLATIONS['period']} ≈ <span id="period-display">—</span> s
                </div>
            </div>

        </div>
        
        <script>
                
            const beforeParams = {json.dumps(before_params)};
            const afterParams = {json.dumps(after_params)};
            const timing = {json.dumps(timing)};
            
            // Translated phase indicators
            const phaseBefore = "{phase_before}";
            const phaseChanging = "{phase_changing}";
            const phaseAfter = "{phase_after}";
            
            // ---- MASS DEFAULTS (added) ----
            if (beforeParams.mass === undefined) beforeParams.mass = 1;
            if (afterParams.mass === undefined)  afterParams.mass  = 1;
            // --------------------------------
            
            const canvas = document.getElementById('pendulum-canvas');
            const ctx = canvas.getContext('2d');
            const originX = 210, originY = 60;
            const baseScale = 80;
            
            let currentParams = {{...beforeParams}};
            let angle = (currentParams.amplitude * Math.PI) / 180;
            let aVel = 0, aAcc = 0;
            const dt = 0.02;
            let startTime = Date.now();
            let phase = 'before';

            function smallAnglePeriod(L, g) {{
                if (L <= 0 || g <= 0) return NaN;
                return 2 * Math.PI * Math.sqrt(L / g);
            }}

            const KN_DIGITS = {{ "0":"೦","1":"೧","2":"೨","3":"೩","4":"೪","5":"೫","6":"೬","7":"೭","8":"೮","9":"೯" }};

            // Format any number/string to Kannada digits; preserves '.', '-', 'e', units, etc.
            function toKannadaNumber(val, decimals = null) {{
                let s;
                if (typeof val === "number" && decimals !== null) {{
                    s = val.toFixed(decimals);            // e.g., 9.81 → "9.81"
                }} else {{
                    s = String(val);
                }}
                return s.replace(/[0-9]/g, d => KN_DIGITS[d]);
            }}

            function updatePhaseIndicator() {{
                const indicator = document.getElementById('phase-indicator');
                const elapsed = (Date.now() - startTime) / 1000;
                
                if (elapsed < timing.before_duration) {{
                    indicator.textContent = phaseBefore;
                    phase = 'before';
                }} else if (elapsed < timing.before_duration + timing.transition_duration) {{
                    indicator.textContent = phaseChanging;
                    phase = 'transition';
                    
                    const transitionProgress = (elapsed - timing.before_duration) / timing.transition_duration;
                    const progress = Math.min(transitionProgress, 1);
                    
                    currentParams.length    = beforeParams.length    + (afterParams.length    - beforeParams.length)    * progress;
                    currentParams.gravity   = beforeParams.gravity   + (afterParams.gravity   - beforeParams.gravity)   * progress;
                    currentParams.amplitude = beforeParams.amplitude + (afterParams.amplitude - beforeParams.amplitude) * progress;
                    currentParams.mass      = beforeParams.mass      + (afterParams.mass      - beforeParams.mass)      * progress;
                    
                    if (progress === 1) {{
                        angle = (currentParams.amplitude * Math.PI) / 180;
                        aVel = 0;
                    }}
                }} else {{
                    indicator.textContent = phaseAfter;
                    phase = 'after';
                    currentParams = {{...afterParams}};
                }}

                document.getElementById('length-display').textContent    =
                toKannadaNumber(currentParams.length, 1) + 'm';

                document.getElementById('gravity-display').textContent   =
                toKannadaNumber(currentParams.gravity, 1) + ' m/s²';

                document.getElementById('amplitude-display').textContent =
                toKannadaNumber(Math.round(currentParams.amplitude), 0) + '°';

                document.getElementById('mass-display').textContent      =
                toKannadaNumber((currentParams.mass ?? 1), 2) + ' kg';

                const T = smallAnglePeriod(currentParams.length, currentParams.gravity);
                const pd = document.getElementById('period-display');
                pd.textContent = isFinite(T) ? toKannadaNumber(T, 2) : '—';

            }}
            
            function drawPendulum() {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                const lengthPixels = currentParams.length * baseScale;
                aAcc = (-currentParams.gravity / currentParams.length) * Math.sin(angle);
                aVel += aAcc * dt;
                aVel *= 0.998;
                angle += aVel * dt;
                
                const bobX = originX + lengthPixels * Math.sin(angle);
                const bobY = originY + lengthPixels * Math.cos(angle);
                
                ctx.beginPath();
                ctx.arc(originX, originY, 6, 0, 2 * Math.PI);
                ctx.fillStyle = '#7c3aed';
                ctx.fill();
                
                ctx.beginPath();
                ctx.moveTo(originX, originY);
                ctx.lineTo(bobX, bobY);
                ctx.strokeStyle = '#7c3aed';
                ctx.lineWidth = 3;
                ctx.stroke();
                
                // ---- MASS VISUALIZATION (added) ----
                // Bob radius grows with mass (visual only); capped for aesthetics
                const m = Math.max(0.1, currentParams.mass || 1);
                const bobRadius = Math.min(12 + 3 * m, 30);
                // ------------------------------------
                
                ctx.beginPath();
                ctx.arc(bobX, bobY, bobRadius, 0, 2 * Math.PI);
                ctx.fillStyle = '#ede9fe';
                ctx.strokeStyle = '#7c3aed';
                ctx.lineWidth = 2.5;
                ctx.fill();
                ctx.stroke();
            }}
            
            function animate() {{
                updatePhaseIndicator();
                drawPendulum();
                requestAnimationFrame(animate);
            }}
            animate();
        </script>
    </body>
    </html>
    """

def display_simulation_if_needed():
    """
    Check if simulation should be displayed and render it.
    Only displays if simulation is active and hasn't been shown for this cycle.
    """
    if (hasattr(st.session_state, 'agent') and 
        st.session_state.agent.state.get("show_simulation")):
        
        simulation_config = st.session_state.agent.state.get("simulation_config")
        
        if simulation_config:
            try:
                # Create and display the simulation
                simulation_html = create_pendulum_simulation_html(simulation_config)
                components.html(simulation_html, height=700, scrolling=True)

                # Add a brief pause instruction using cached translation
                st.info("🔬 " + UI_TRANSLATIONS['simulation_running_msg'])

                # Mark simulation as displayed but keep it available for this cycle
                # We don't reset show_simulation here - let the nodes manage the lifecycle
                
            except Exception as e:
                st.error(f"Error displaying simulation: {e}")
                # Clear flags on error
                st.session_state.agent.state["show_simulation"] = False
                st.stop()

def display_image_with_context(image_data, show_explanation=True):
    """Enhanced image display with educational context"""

    print("reached here,IMAGE SHOULD print")
    
    # Main image with responsive sizing
    st.image(
        image_data["url"],
        caption=f"{image_data['description']}",
        use_container_width=False
    )
    
    # Optional: Educational context in an expander
    if show_explanation and image_data.get("relevance_reason"):
        with st.expander(f" {UI_TRANSLATIONS['why_image_helps']}"):
            st.write(image_data["relevance_reason"])
            
    # Add a subtle divider after image
    st.markdown("---")


def render_viseme_sidebar(latest_text: str, key: str = "viseme_iframe"):
    latest_text = (latest_text or "").strip()

    def js_escape(s):
        return s.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${}")
    safe_text = js_escape(latest_text)

    # Pull the most recent audio's message id (so we sync only to the latest)
    current_msg_id = st.session_state.get('latest_audio_msg_id', "")

    boy_image_path = os.path.join("static", "myphoto2.png")
    girl_image_path = os.path.join("static", "myphoto.png")
    boy_image_b64 = get_image_base64(boy_image_path)
    girl_image_b64 = get_image_base64(girl_image_path)
    if not boy_image_b64 or not girl_image_b64:
        st.error("Could not load character images from static folder")
        return

    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width,initial-scale=1" />
      <style>
        body {{ background: transparent; color: #eee; font-family: Arial, sans-serif; margin: 0; padding: 0; }}
        .container {{ padding: 8px 6px 2px 6px; }}
        .stage {{ position:relative; display:block; }}
        .character {{ width:100%; max-width:260px; display:block; margin:0 auto; position: relative; z-index: 1; }}
        .mouth-container {{ position:absolute; left:25px; top:145px; width:151px; height:36px; pointer-events:none; z-index: 2; }}
        svg.mouth {{ width:100%; height:100%; }}
        .mouth-set g {{ opacity:0; transition:opacity 80ms ease-out; }}
        .mouth-set g.active {{ opacity:1; }}
        .status {{ margin:6px 0 2px 0; text-align:center; font-size:12px; color:#bbb; }}
      </style>
    </head>
    <body>
      <div class="container">
        <div class="stage">
          <img id="characterImage" src="data:image/png;base64,{boy_image_b64}" class="character" />
          <div class="mouth-container">
            <svg class="mouth" viewBox="-50 -50 100 100">
              <g class="mouth-set" id="mouthSet">
                <g data-viseme="rest" class="active"><path d="M-26 6 q26 10 52 0" fill="#c33" stroke="#000" stroke-width="1"/></g>
                <g data-viseme="closed"><rect x="-30" y="-6" width="60" height="12" rx="6" fill="#9b2b2b"/></g>
                <g data-viseme="wide"><ellipse cx="0" cy="6" rx="42" ry="14" fill="#9b2b2b"/></g>
                <g data-viseme="open"><ellipse cx="0" cy="8" rx="36" ry="22" fill="#9b2b2b"/></g>
                <g data-viseme="round"><ellipse cx="0" cy="6" rx="22" ry="26" fill="#9b2b2b"/></g>
                <g data-viseme="f_v">
                  <path d="M-28 6 q28 -24 56 0" fill="none" stroke="#000" stroke-width="3" />
                  <rect x="-20" y="2" width="40" height="6" rx="3" fill="#9b2b2b" />
                </g>
                <g data-viseme="th">
                  <rect x="-18" y="0" width="36" height="8" rx="4" fill="#9b2b2b" />
                  <rect x="-6" y="-8" width="12" height="8" rx="3" fill="#ffe8d6" />
                </g>
                <g data-viseme="smush"><ellipse cx="0" cy="6" rx="28" ry="12" fill="#9b2b2b"/></g>
                <g data-viseme="kiss"><ellipse cx="0" cy="6" rx="16" ry="12" fill="#9b2b2b"/></g>
              </g>
            </svg>
          </div>
        </div>

        <div class="status">
          <div>Viseme: <strong id="visemeName">rest</strong></div>
        </div>
      </div>

      <script>
        const INJECTED_TEXT = `{safe_text}`;
        const CURRENT_MSG_ID = `{current_msg_id}`;

        const mouthSet  = document.getElementById("mouthSet");
        const visemeLbl = document.getElementById("visemeName");

        // ----- text → frames (your precompute, kept) -----
        function simpleG2P(text) {{
            let s = (text || "").toLowerCase().replace(/[^a-z\s]/g, ' ');
            const tokens = [];
            for (let i=0;i<s.length;) {{
            if (s[i]===" ") {{ i++; continue }}
            const dig = s.slice(i,i+2);
            if (['ch','sh','th','ng','ph','qu','ck','wh'].includes(dig)) {{ tokens.push(dig); i+=2; continue }}
            tokens.push(s[i]); i++;
            }}
            return tokens;
        }}
        function phonemeToViseme(p) {{
            if (['p','b','m'].includes(p))      return 'closed';
            if (['a','o'].includes(p))          return 'open';
            if (['e','i','y'].includes(p))      return 'wide';
            if (['u','oo','w'].includes(p))     return 'round';
            if (['f','v'].includes(p))          return 'f_v';
            if (['th','t','d','n'].includes(p)) return 'th';
            if (['s','z','sh','ch','j'].includes(p)) return 'smush';
            return 'rest';
        }}
        function estimateDur(tok) {{ return /[aeiou]/.test(tok) ? 140 : 90; }} // ms

        function buildFrames(text) {{
            const toks   = simpleG2P(text);
            const frames = toks.map(t => ({{ vis: phonemeToViseme(t), dur: estimateDur(t) }}));
            // merge adjacent identical visemes
            const merged = [];
            for (const f of frames) {{
            const last = merged[merged.length - 1];
            if (last && last.vis === f.vis) last.dur += f.dur; else merged.push({{ ...f }});
            }}
            let cum = 0;
            for (const f of merged) {{ f.start = cum; cum += f.dur; f.end = cum; }}
            merged.totalTextMs = cum;
            return merged;
        }}

        // ----- render helper -----
        function setViseme(v) {{
            mouthSet.querySelectorAll("[data-viseme]").forEach(g => g.classList.remove("active"));
            const el = mouthSet.querySelector(`[data-viseme="${{v}}"]`);
            if (el) el.classList.add("active");
            visemeLbl.innerText = v;
        }}

        // Initial state
        document.addEventListener('DOMContentLoaded', () => {{
            const el = mouthSet.querySelector('[data-viseme="rest"]');
            if (el) el.classList.add('active');
            visemeLbl.innerText = 'rest';
        }});

        // ----- time-driven mapping -----
        let frames     = buildFrames(INJECTED_TEXT);
        let audioDurMs = null;

        function visemeAtTime(tMs) {{
            if (!frames || !frames.length) return 'rest';
            const totalText = frames.totalTextMs || 1;
            const dur = (audioDurMs && audioDurMs > 0) ? audioDurMs : (totalText / 1.25); // fallback if no duration yet
            const progress = Math.max(0, Math.min(1, (tMs || 0) / dur));
            const targetText = progress * totalText;

            // binary search frame whose [start, end) contains targetText
            let lo = 0, hi = frames.length - 1, ans = 0;
            while (lo <= hi) {{
            const mid = (lo + hi) >> 1;
            const f = frames[mid];
            if (targetText < f.start) hi = mid - 1;
            else if (targetText >= f.end) lo = mid + 1;
            else {{ ans = mid; break; }}
            ans = Math.min(Math.max(ans, 0), frames.length - 1);
            }}
            return frames[ans].vis || 'rest';
        }}

        // ----- messages from the audio iframe -----
        const bc = new BroadcastChannel('agent_audio');
        bc.onmessage = (ev) => {{
        const d = ev.data || {{}};
        if (!d.type) return;
        if (d.id && d.id !== CURRENT_MSG_ID) return; // only react to the latest message

        if (d.type === 'audio_meta') {{
            if (d.dur && d.dur > 0) audioDurMs = d.dur;
        }}
        if (d.type === 'audio_play') {{
            audioDurMs = d.dur || audioDurMs;
            setViseme(visemeAtTime(d.t || 0)); // start exactly with audio
        }}
        if (d.type === 'audio_tick') {{
            audioDurMs = d.dur || audioDurMs;
            setViseme(visemeAtTime(d.t || 0)); // follow the audio clock
        }}
        if (d.type === 'audio_pause' || d.type === 'audio_end') {{
            setViseme('rest');
        }}
    }};
        </script>

    </body>
    </html>
    """
    with st.sidebar:
        components.html(html, height=330, scrolling=False)



# ── Streamlit Page Configuration & State Initialization ──────────────────
st.set_page_config(page_title="Interactive Educational Agent", page_icon="🤖")
if "audio_rendered_for_ids" not in st.session_state:
    st.session_state.audio_rendered_for_ids = set()
if "original_english_texts" not in st.session_state:
    st.session_state.original_english_texts = {}

def generate_session_id():
    """Generate a unique session ID for Langfuse tracking"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"streamlit-user-{timestamp}"

def initialize_agent():
    """Initialize the Educational Agent with Langfuse session tracking"""
    session_id = generate_session_id()
    agent = EducationalAgent(
        session_label="streamlit-simulation-session",
        user_id="streamlit-user",
        persona_name="interactive-user"
    )
    return agent, session_id



if "session_started" not in st.session_state:
    # Use cached translations
    st.title(f"🧑‍🎓 {UI_TRANSLATIONS['interactive_simulation_educational_agent']}")
    
    # Format welcome message with concept title
    welcome_text = UI_TRANSLATIONS['welcome_msg_template'].format(concept=concept_pkg.title)
    st.info(welcome_text)
    
    if st.button(f"🚀 {UI_TRANSLATIONS['start_learning']}", type="primary"):
        # Initialize the agent and session
        agent, session_id = initialize_agent()
        
        st.session_state.session_started = True
        st.session_state.agent = agent
        st.session_state.session_id = session_id
        st.session_state.messages = []
        st.session_state.audio_recorder_key_counter = 0
        st.session_state.processing_request = True  # Trigger initial processing for welcome message
        
        st.rerun()
    st.stop()

# ── Step 2: Process a request if the flag is set ───────────────────────────
if st.session_state.get("processing_request"):
    st.session_state.processing_request = False  # Unset flag to prevent re-running

    # Use a spinner during the potentially slow LLM call
    with st.spinner(f"🤔 {UI_TRANSLATIONS['thinking']}"):
        try:
            # Check if this is the initial start
            if not st.session_state.messages:
                # Start the conversation
                agent_reply = st.session_state.agent.start()
            else:
                # Continue the conversation with the last user message
                last_user_msg = None
                for message_data in reversed(st.session_state.messages):
                    # Handle both old and new message formats
                    if len(message_data) == 2:
                        role, msg = message_data
                    else:
                        role, msg, _ = message_data
                    
                    if role == "user":
                        last_user_msg = msg
                        break
                
                if last_user_msg:
                    agent_reply = st.session_state.agent.post(last_user_msg)
                else:
                    agent_reply = "I'm waiting for your response."
            
            if agent_reply:
                # Store original English text for stable hashing
                message_index = len(st.session_state.messages)
                st.session_state.original_english_texts[message_index] = agent_reply
                print(f"📝 STORED original English for message {message_index}: '{agent_reply[:50]}...'")
                
                # Check if there's enhanced metadata from the agent
                if (hasattr(st.session_state.agent, 'state') and 
                    st.session_state.agent.state.get("enhanced_message_metadata")):
                    metadata = st.session_state.agent.state["enhanced_message_metadata"]
                    st.session_state.messages.append(("assistant", agent_reply, metadata))
                    # Clear the metadata after use to prevent duplication
                    st.session_state.agent.state.pop("enhanced_message_metadata", None)
                else:
                    st.session_state.messages.append(("assistant", agent_reply))
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            agent_reply = "I encountered an error. Please try again."
            st.session_state.messages.append(("assistant", agent_reply))
            st.stop()
    
    st.rerun()

# ── Main Application Logic & UI Display ──────────────────────────────────
st.title(f"🧑‍🎓 {UI_TRANSLATIONS['interactive_simulation_educational_agent']}")

# Display session info in sidebar
with st.sidebar:
    # 1) VISeme at the very top (latest assistant message)
    last_assistant_text = None
    for message_data in reversed(st.session_state.get("messages", [])):
        # Handle both old and new message formats
        if len(message_data) == 2:
            role, msg = message_data
        else:
            role, msg, _ = message_data
        
        if role == "assistant" and isinstance(msg, str) and msg.strip():
            last_assistant_text = msg
            break

    if last_assistant_text:
        # Find the latest assistant message index to get original English
        latest_assistant_index = None
        for idx, message_data in enumerate(reversed(st.session_state.get("messages", []))):
            if len(message_data) >= 2 and message_data[0] == "assistant":
                latest_assistant_index = len(st.session_state.messages) - 1 - idx
                break
        
        # Use original English for hash generation
        if latest_assistant_index is not None:
            original_english = st.session_state.original_english_texts.get(latest_assistant_index, last_assistant_text)
        else:
            original_english = last_assistant_text
            
        viseme_hash = msg_id_from_text(original_english)
        st.session_state['latest_audio_msg_id'] = viseme_hash
        print(f"👄 VISEME HASH: '{viseme_hash[:12]}...' from ORIGINAL English: '{original_english[:50]}...'")
        print(f"👄 (Viseme will process English: '{original_english[:30]}...')")


    # Use original English for viseme processing (no translation needed)
    if latest_assistant_index is not None:
        original_english_for_viseme = st.session_state.original_english_texts.get(latest_assistant_index, last_assistant_text)
        print(f"👄 Using ORIGINAL English for viseme: '{original_english_for_viseme[:50]}...'")
        last_assistant_text = original_english_for_viseme
    else:
        print(f"⚠️ No original English found, using current text: '{last_assistant_text[:50]}...'")

    print(f"👄 Final viseme text: '{last_assistant_text[:50]}...'")
    print(f"📊 Hash comparison - Both should now match: {st.session_state.get('latest_audio_msg_id', 'None')[:12] if st.session_state.get('latest_audio_msg_id') else 'None'}...")
    # Render the character (auto-plays on each new assistant msg)
    render_viseme_sidebar(last_assistant_text, key="viseme_iframe_top")

    # 2) The rest of your existing sidebar content
    st.header(f"📊 {UI_TRANSLATIONS['session_info']}")
    if "agent" in st.session_state:
        session_info = st.session_state.agent.session_info()
        st.write(f"**Session ID:** {session_info['session_id']}")
        st.write(f"**User ID:** {session_info['user_id']}")
        st.write(f"**Current State:** {st.session_state.agent.current_state()}")
        st.write(f"**Concept:** {concept_pkg.title}")
        if session_info.get('tags'):
            st.write(f"**Tags:** {session_info['tags']}")

    st.markdown("---")
    st.markdown(f"**💡 {UI_TRANSLATIONS['how_to_interact']}**")
    st.markdown(f"- {UI_TRANSLATIONS['type_responses']}")
    st.markdown(f"- {UI_TRANSLATIONS['use_microphone']}")
    st.markdown(f"- {UI_TRANSLATIONS['agent_guide']}")


# Display all messages. The audio player is only added for the last assistant message.
for i, message_data in enumerate(st.session_state.messages):
    # Handle both old format (role, content) and new format (role, content, metadata)
    if len(message_data) == 2:  # Old format
        role, msg = message_data
        metadata = {}
    else:  # New format with metadata
        role, msg, metadata = message_data
    
    # Check if message needs translation and update session state
    try:
        detected_lang = detect(msg)
        if detected_lang == 'en':
            print(f"🔄 TRANSLATING message {i+1}: '{msg[:30]}...' (detected: {detected_lang})")
            translated_msg = GoogleTranslator(source='en', target='kn').translate(msg)
            # Update the message in session state with the translated version
            if len(message_data) == 2:  # Old format
                st.session_state.messages[i] = (role, translated_msg)
            else:  # New format with metadata
                st.session_state.messages[i] = (role, translated_msg, metadata)
            msg = translated_msg
            print(f"✅ TRANSLATED to: '{msg[:30]}...'")
        else:
            print(f"ℹ️ No translation needed for message {i+1}: '{msg[:30]}...' (detected: {detected_lang})")
    except Exception as e:
        # If language detection or translation fails, use original message
        st.warning(f"Translation failed for message {i+1}: {e}")
        pass
    
    with st.chat_message(role):
        st.write(msg)
        
        # Display image if present in metadata for assistant messages
        if role == "assistant" and metadata.get("image"):
            display_image_with_context(metadata["image"])
        
        # Check if we need to show simulation after this assistant message
        if role == "assistant" and (i == len(st.session_state.messages) - 1):
            # Display simulation if needed
            display_simulation_if_needed()
            
            # Add audio playback for the latest assistant message (only once per message)
            try:
                # Use original English text for stable hashing
                original_english = st.session_state.original_english_texts.get(i, msg)
                mid = msg_id_from_text(original_english)
                print(f"🔊 AUDIO HASH: '{mid[:12]}...' from ORIGINAL English: '{original_english[:50]}...'")
                print(f"🔊 (Audio will play Kannada: '{msg[:30]}...')")

                if mid not in st.session_state.audio_rendered_for_ids:
                    # First render for this assistant reply → create <audio autoplay> + postMessage hooks
                    returned_id = play_text_as_audio(
                        msg,
                        st.container(),
                        message_id=mid,          # keep the id stable across reruns
                        speed_factor=1.25        # your audio speed-up
                    )
                    st.session_state.audio_rendered_for_ids.add(mid)
                    st.session_state['latest_audio_msg_id'] = returned_id  # same as mid
                else:
                    # Already rendered this audio; DO NOT create a new <audio> element (prevents restart)
                    # Just keep the id for the viseme sidebar to know what to sync to
                    st.session_state['latest_audio_msg_id'] = mid

            except Exception as e:
                st.caption("⚠️ Audio playback unavailable")
                st.stop()

            # pass

# Handle user input at the bottom of the page
if "agent" in st.session_state and st.session_state.agent.current_state() != "END":
    user_msg = None
    
    # Audio input
    col1, col2 = st.columns([3, 1])
    with col2:
        st.caption(f"🎤 {UI_TRANSLATIONS['voice_input']}")
        recorded_audio_bytes = audio_recorder(
            text=UI_TRANSLATIONS['click_to_speak'],
            key=f"audio_recorder_{st.session_state.audio_recorder_key_counter}",
            icon_size="1x", 
            pause_threshold=2.0
        )
        
    if recorded_audio_bytes:
        with st.spinner(f"🎯 {UI_TRANSLATIONS['transcribing']}"):
            user_msg = transcribe_recorded_audio_bytes(recorded_audio_bytes)
            if user_msg and not user_msg.startswith("["):  # Valid transcription
                st.success(f"{UI_TRANSLATIONS['you_said']} {user_msg}")

    # Text input
    text_input = st.chat_input(f"💬 {UI_TRANSLATIONS['type_response_here']}")
    if text_input:
        user_msg = text_input

    # ── Step 1: Acknowledge user input and trigger the "Safe State" rerun ──
    if user_msg and not user_msg.startswith("["):  # Valid user input
        st.session_state.audio_recorder_key_counter += 1
        
        # Add user's message and set the flag to process it on the next run
        st.session_state.messages.append(("user", user_msg))
        st.session_state.processing_request = True
        
        # This rerun is fast. It redraws the page without the old audio player.
        st.rerun()

# Session End Summary
if "agent" in st.session_state and st.session_state.agent.current_state() == "END":
    st.markdown("---")
    st.success(f"🎉 {UI_TRANSLATIONS['learning_session_complete']}")
    
    # Get session summary from agent state
    session_summary = st.session_state.agent.state.get("session_summary", {})
    
    if session_summary:
        st.subheader(f"📋 {UI_TRANSLATIONS['session_summary']}")
        st.json(session_summary)
        
        # Download session summary
        summary_json = json.dumps(session_summary, indent=2)
        st.download_button(
            label=f"📥 {UI_TRANSLATIONS['download_session_summary']}", 
            data=summary_json, 
            file_name=f"session_summary_{st.session_state.session_id}.json", 
            mime="application/json"
        )
    else:
        st.info(UI_TRANSLATIONS['session_completed'])
    
    # Show session info for Langfuse tracking
    if "agent" in st.session_state:
        session_info = st.session_state.agent.session_info()
        st.subheader(f"🔍 {UI_TRANSLATIONS['langfuse_session_details']}")
        st.code(f"Session ID: {session_info['session_id']}\nThread ID: {session_info['thread_id']}")
    
    # Compute and upload session metrics
    if "session_metrics_computed" not in st.session_state:
        with st.spinner(f"📊 {UI_TRANSLATIONS['computing_session_metrics']}"):
            try:
                # Convert messages to history format for metrics
                history_for_reports = st.session_state.agent.get_history_for_reports()
                
                session_metrics = compute_and_upload_session_metrics(
                    session_id=st.session_state.agent.session_id,
                    history=history_for_reports,
                    session_state=st.session_state.agent.state,
                    persona_name="interactive-user"
                )
                st.session_state.session_metrics = session_metrics
                st.session_state.session_metrics_computed = True
                st.success(f"✅ {UI_TRANSLATIONS['session_metrics_computed']}")
            except Exception as e:
                st.error(f"❌ {UI_TRANSLATIONS['failed_compute_metrics']} {e}")
                st.session_state.session_metrics_computed = True  # Mark as attempted to avoid retry
    
    # Display computed metrics
    if "session_metrics" in st.session_state:
        st.subheader(f"📊 {UI_TRANSLATIONS['session_metrics']}")
        metrics = st.session_state.session_metrics
        
        # Key metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(UI_TRANSLATIONS['quiz_score'], f"{metrics.quiz_score:.1f}%")
            st.metric(UI_TRANSLATIONS['user_type'], metrics.user_type)
        with col2:
            st.metric(UI_TRANSLATIONS['engagement_rating'], f"{metrics.user_engagement_rating:.1f}/5")
            st.metric(UI_TRANSLATIONS['interest_rating'], f"{metrics.user_interest_rating:.1f}/5")
        with col3:
            st.metric(UI_TRANSLATIONS['concepts_covered'], metrics.num_concepts_covered)
            st.metric(UI_TRANSLATIONS['enjoyment_probability'], f"{metrics.enjoyment_probability:.0%}")
        
        # Download metrics
        metrics_json = metrics.model_dump_json(indent=2)
        st.download_button(
            label=f"📊 {UI_TRANSLATIONS['download_session_metrics']}",
            data=metrics_json,
            file_name=f"session_metrics_{st.session_state.session_id}.json",
            mime="application/json"
        )
    
    # Option to start a new session
    if st.button(f"🔄 {UI_TRANSLATIONS['start_new_session']}", type="primary"):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Footer
st.markdown("---")
st.caption(f"🤖 {UI_TRANSLATIONS['powered_by']} | 📊 {UI_TRANSLATIONS['tracked_with']}")
