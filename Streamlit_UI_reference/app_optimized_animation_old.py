import os
import streamlit as st
import streamlit.components.v1 as components
import json
import onnx_asr
from scipy.io import wavfile
import numpy as np
import tempfile
import base64
import time
import soundfile as sf
from pedalboard import Pedalboard, Resample
import sys
import pysqlite3
from datetime import datetime
from dotenv import load_dotenv

# Import the audio_recorder component
from audio_recorder_streamlit import audio_recorder

# Import gTTS for text-to-speech
from gtts import gTTS

sys.modules["sqlite3"] = pysqlite3

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

class WhisperASR:
    def __init__(self, model_name: str = "tiny"):
        # Load tiny (~75MB). Use "base", "small", etc. if you want better accuracy.
        self.model = whisper.load_model(model_name)

    def recognize(self, audio_path: str) -> str:
        # fp16=False is safer on CPU; set True on GPU with half precision.
        result = self.model.transcribe(audio_path, language='en', fp16=False)
        return result.get("text", "").strip()


# Load environment variables
load_dotenv(dotenv_path=".env", override=True)

# Import the EducationalAgent class
try:
    from educational_agent_optimized_langsmith.agent import EducationalAgent
    from educational_agent_optimized_langsmith.config import concept_pkg
    from tester_agent.session_metrics import compute_and_upload_session_metrics
except ImportError as e:
    st.error(f"Could not import EducationalAgent: {e}")
    st.stop()
    
# ── ASR & TTS Functions ──────────────────────────────────────────────────
@st.cache_resource(ttl=36000)
def load_asr_model():
    print("BOOT: about to init ASR...", flush=True)
    # model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2")
    model = WhisperASR(model_name="small")
    print("BOOT: ASR ready", flush=True)
    return model
    # return None
    # return onnx_asr.load_model(model = "nemo-parakeet-tdt-0.6b-v2", path = "parakeet-tdt-0.6b-v2-onnx")

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

def play_text_as_audio(text, container):
    """
    Generates audio with gTTS and speeds it up using pedalboard, saving as WAV.
    """
    if not text or not text.strip():
        return
    
    try:
        # 1. gTTS generates the initial MP3 audio
        tts = gTTS(text=text, lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            normal_speed_path = fp.name

        # 2. Load the audio file
        audio, sample_rate = sf.read(normal_speed_path)

        # 3. Create a pedalboard to resample (speed up) the audio
        board = Pedalboard([
            Resample(target_sample_rate=int(sample_rate * 1.25))
        ])
        
        # 4. Process the audio
        fast_audio = board(audio, sample_rate)
        
        # 5. Export the fast audio to a temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            sf.write(fp.name, fast_audio, int(sample_rate * 1.25), format='WAV')
            fast_speed_path = fp.name

        # 6. Read bytes and encode for Streamlit
        with open(fast_speed_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        # 7. Clean up temporary files
        os.remove(normal_speed_path)
        os.remove(fast_speed_path)

        # 8. Display in Streamlit using the correct audio type
        audio_html = f"""
        <audio controls autoplay style="width: 100%; margin-top: 5px;">
            <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
        </audio>
        """
        container.markdown(audio_html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred in audio processing: {e}")

# ── Simulation Integration Functions ──────────────────────────────────────
def create_pendulum_simulation_html(config):
    """
    Generate HTML for pendulum simulation based on configuration.
    Uses the simulation from index.html with automated before/after demonstration.
    """
    before_params = config['before_params']
    after_params = config['after_params']
    timing = config['timing']
    agent_message = config['agent_message']
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .simulation-container {{
                width: 100%;
                max-width: 600px;
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
        </style>
    </head>
    <body>
        <div class="simulation-container">
            <div class="agent-message">{agent_message}</div>
            <div id="phase-indicator" class="phase-indicator">Phase: Before Change</div>
            
            <canvas id="pendulum-canvas" class="simulation-canvas" width="400" height="300"></canvas>
            
            <div class="simulation-controls">
                <div class="param-display">
                    Length: <span id="length-display">{before_params['length']:.1f}m</span>
                </div>
                <div class="param-display">
                    Gravity: <span id="gravity-display">{before_params['gravity']:.1f} m/s²</span>
                </div>
                <div class="param-display">
                    Amplitude: <span id="amplitude-display">{before_params['amplitude']}°</span>
                </div>
            </div>
        </div>
        
        <script>
            // Simulation parameters
            const beforeParams = {json.dumps(before_params)};
            const afterParams = {json.dumps(after_params)};
            const timing = {json.dumps(timing)};
            
            // Canvas setup
            const canvas = document.getElementById('pendulum-canvas');
            const ctx = canvas.getContext('2d');
            const originX = 200, originY = 60;
            const baseScale = 80;
            
            // Animation state
            let currentParams = {{...beforeParams}};
            let angle = (currentParams.amplitude * Math.PI) / 180;
            let aVel = 0, aAcc = 0;
            const dt = 0.02;
            let startTime = Date.now();
            let phase = 'before'; // 'before', 'transition', 'after'
            
            function updatePhaseIndicator() {{
                const indicator = document.getElementById('phase-indicator');
                const elapsed = (Date.now() - startTime) / 1000;
                
                if (elapsed < timing.before_duration) {{
                    indicator.textContent = 'Phase: Before Change';
                    phase = 'before';
                }} else if (elapsed < timing.before_duration + timing.transition_duration) {{
                    indicator.textContent = 'Phase: Changing Parameters...';
                    phase = 'transition';
                    
                    // Smooth transition
                    const transitionProgress = (elapsed - timing.before_duration) / timing.transition_duration;
                    const progress = Math.min(transitionProgress, 1);
                    
                    // Interpolate parameters
                    currentParams.length = beforeParams.length + (afterParams.length - beforeParams.length) * progress;
                    currentParams.gravity = beforeParams.gravity + (afterParams.gravity - beforeParams.gravity) * progress;
                    currentParams.amplitude = beforeParams.amplitude + (afterParams.amplitude - beforeParams.amplitude) * progress;
                    
                    // Reset pendulum when transition completes
                    if (progress === 1) {{
                        angle = (currentParams.amplitude * Math.PI) / 180;
                        aVel = 0;
                    }}
                }} else {{
                    indicator.textContent = 'Phase: After Change';
                    phase = 'after';
                    currentParams = {{...afterParams}};
                }}
                
                // Update displays
                document.getElementById('length-display').textContent = currentParams.length.toFixed(1) + 'm';
                document.getElementById('gravity-display').textContent = currentParams.gravity.toFixed(1) + ' m/s²';
                document.getElementById('amplitude-display').textContent = Math.round(currentParams.amplitude) + '°';
            }}
            
            function drawPendulum() {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // Calculate pendulum physics
                const lengthPixels = currentParams.length * baseScale;
                aAcc = (-currentParams.gravity / currentParams.length) * Math.sin(angle);
                aVel += aAcc * dt;
                aVel *= 0.998; // Damping
                angle += aVel * dt;
                
                // Calculate bob position
                const bobX = originX + lengthPixels * Math.sin(angle);
                const bobY = originY + lengthPixels * Math.cos(angle);
                
                // Draw pivot point
                ctx.beginPath();
                ctx.arc(originX, originY, 6, 0, 2 * Math.PI);
                ctx.fillStyle = '#7c3aed';
                ctx.fill();
                
                // Draw string
                ctx.beginPath();
                ctx.moveTo(originX, originY);
                ctx.lineTo(bobX, bobY);
                ctx.strokeStyle = '#7c3aed';
                ctx.lineWidth = 3;
                ctx.stroke();
                
                // Draw bob
                ctx.beginPath();
                ctx.arc(bobX, bobY, 15, 0, 2 * Math.PI);
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
            
            // Start animation
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
                components.html(simulation_html, height=450)
                
                # Add a brief pause instruction
                st.info("🔬 **Simulation running above** - Watch the pendulum carefully and notice what changes!")
                
                # Mark simulation as displayed but keep it available for this cycle
                # We don't reset show_simulation here - let the nodes manage the lifecycle
                
            except Exception as e:
                st.error(f"Error displaying simulation: {e}")
                # Clear flags on error
                st.session_state.agent.state["show_simulation"] = False
                st.stop()

def render_viseme_sidebar(latest_text: str, key: str = "viseme_iframe"):
    """
    Renders the viseme character (from animation.html) in the sidebar and auto-plays
    SpeechSynthesis for `latest_text`, with lip-sync.

    Notes:
      - Uses browser SpeechSynthesis (no gTTS for this playback).
      - Auto plays whenever this component is re-rendered with new text.
      - Minimal UI: character + small status; voice & char selectors kept compact.
    """
    # Fallback text if empty/None
    latest_text = (latest_text or "").strip()
    # Escape for JS
    def js_escape(s):
        return s.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${}")
    safe_text = js_escape(latest_text)
    
    # Load and encode images
    boy_image_path = os.path.join("static", "myphoto2.png")
    girl_image_path = os.path.join("static", "myphoto.png")
    boy_image_b64 = get_image_base64(boy_image_path)
    girl_image_b64 = get_image_base64(girl_image_path)
    
    if not boy_image_b64 or not girl_image_b64:
        st.error("Could not load character images from static folder")
        return

    # --- Compact HTML adapted from your animation.html with small CSS tweaks ---
    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width,initial-scale=1" />
      <style>
        body {{
          background: transparent; color: #eee; font-family: Arial, sans-serif;
          margin: 0; padding: 0;
        }}
        .container {{
          padding: 8px 6px 2px 6px;
        }}
        .stage {{ position:relative; display:block; }}
        .character {{ width:100%; max-width:260px; display:block; margin:0 auto; }}
        .mouth-container {{
          position:absolute; left:25px; top:145px; width:151px; height:36px; pointer-events:none;
        }}
        svg.mouth {{ width:100%; height:100%; }}
        .mouth-set g {{ opacity:0; transition:opacity 80ms ease-out; }}
        .mouth-set g.active {{ opacity:1; }}
        .status {{ margin:6px 0 2px 0; text-align:center; font-size:12px; color:#bbb; }}
        .row {{ display:flex; align-items:center; gap:6px; justify-content:center; }}
        select {{
          background:#222; color:#eee; border:1px solid #555; padding:4px 6px; border-radius:4px; font-size:12px;
          max-width: 180px;
        }}
        .character-selector button {{
          background:#444; color:#eee; border:1px solid #666; padding:3px 8px; border-radius:10px; cursor:pointer;
          font-size:12px;
        }}
        .character-selector button.active {{ background:#0066cc; border-color:#0088ff; }}
        .controls, #textInput, #playBtn, #stopBtn {{ display:none; }} /* hide manual controls */
      </style>
    </head>
    <body>
      <div class="container">
        <div class="stage">
          <img id="characterImage" src="data:image/png;base64,{boy_image_b64}" class="character" />
          <div class="mouth-container">
            <svg class="mouth" viewBox="-50 -50 100 100">
              <g class="mouth-set" id="mouthSet">
                <g data-viseme="rest"><path d="M-26 6 q26 10 52 0" fill="#c33" stroke="#000" stroke-width="1"/></g>
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
          <div class="row">
            <span id="currentCharacterName">Boy</span> ·
            <span>Voice:</span>
            <select id="voiceSelect"></select>
          </div>
          <div>Viseme: <strong id="visemeName">rest</strong></div>
        </div>

        <!-- Hidden controls we reuse -->
        <textarea id="textInput" rows="2" cols="20"></textarea>
        <div class="controls">
          <button id="playBtn">Play</button>
          <button id="stopBtn">Stop</button>
        </div>
      </div>

      <script>
        const visemeName = document.getElementById("visemeName");
        const mouthSet = document.getElementById("mouthSet");
        const textInput = document.getElementById("textInput");
        const voiceSelect = document.getElementById("voiceSelect");
        const characterImage = document.getElementById("characterImage");
        const currentCharacterName = document.getElementById("currentCharacterName");

        let queue = [], timer=null, playing=false, forceStop=false;
        let utterance = null;
        let voices = [];
        let selectedVoice = null;
        let currentCharacter = 'boy';

        // Character assets (same as your HTML)
        const characters = {{
          boy: {{ name:'Boy', image:'data:image/png;base64,{boy_image_b64}', preferMale: true }},
          girl: {{ name:'Girl', image:'data:image/png;base64,{girl_image_b64}', preferMale: false }}
        }};

        // Voice loading
        function loadVoicesForCharacter() {{
          const preferMale = characters[currentCharacter].preferMale;
          voiceSelect.innerHTML = '';
          let all = speechSynthesis.getVoices();
          voices = all;

          let filtered = all.filter(v => {{
            const name = v.name.toLowerCase();
            const maleMarkers = ['male','man','boy','david','mark','alex','daniel','james','thomas','richard'];
            const femaleMarkers = ['female','woman','girl','karen','samantha','victoria','zira','susan','anna','lily','emma','sophia'];
            const markerHit = (arr)=>arr.some(m=>name.includes(m));
            return preferMale ? markerHit(maleMarkers) : markerHit(femaleMarkers);
          }});
          if (filtered.length === 0) filtered = all;

          filtered.forEach(voice => {{
            const opt = document.createElement('option');
            opt.value = all.indexOf(voice);
            opt.textContent = voice.name + ' (' + voice.lang + ')';
            voiceSelect.appendChild(opt);
          }});

          selectedVoice = filtered[0] || all[0] || null;
          if (selectedVoice) voiceSelect.value = all.indexOf(selectedVoice);
        }}

        function loadVoices() {{
          loadVoicesForCharacter();
        }}
        speechSynthesis.onvoiceschanged = loadVoices;
        loadVoices();

        voiceSelect.onchange = function() {{
          const idx = parseInt(this.value);
          selectedVoice = speechSynthesis.getVoices()[idx] || null;
        }};

        function simpleG2P(text) {{
          let s = text.toLowerCase().replace(/[^a-z\\s]/g, ' ');
          const tokens = [];
          for (let i=0;i<s.length;) {{
            if (s[i]===" ") {{ i++; continue }}
            const dig=s.slice(i,i+2);
            if (['ch','sh','th','ng','ph','qu','ck','wh'].includes(dig)) {{ tokens.push(dig); i+=2; continue }}
            tokens.push(s[i]); i++;
          }}
          return tokens;
        }}

        function phonemeToViseme(p) {{
          if (['p','b','m'].includes(p)) return 'closed';
          if (['a','o'].includes(p)) return 'open';
          if (['e','i','y'].includes(p)) return 'wide';
          if (['u','oo','w'].includes(p)) return 'round';
          if (['f','v'].includes(p)) return 'f_v';
          if (['th','t','d','n'].includes(p)) return 'th';
          if (['s','z','sh','ch','j'].includes(p)) return 'smush';
          if (['q'].includes(p)) return 'kiss';
          return 'rest';
        }}

        function estimateDur(tok) {{ return /[aeiou]/.test(tok) ? 140 : 90; }}

        function prepareQueue(text) {{
          const toks = simpleG2P(text);
          const frames = toks.map(t=>({{vis:phonemeToViseme(t), dur:estimateDur(t)}}));
          const comp=[];
          for (const f of frames) {{
            const last=comp[comp.length-1];
            if (last && last.vis===f.vis) last.dur+=f.dur;
            else comp.push({{...f}});
          }}
          return comp;
        }}

        function setViseme(v) {{
          mouthSet.querySelectorAll("[data-viseme]").forEach(g=>g.classList.remove("active"));
          const el=mouthSet.querySelector(`[data-viseme="${{v}}"]`);
          if (el) el.classList.add("active");
          visemeName.innerText=v;
        }}

        function stepQueue() {{
          if (!playing || forceStop) return;
          if (queue.length===0) {{ setViseme("rest"); playing=false; return }}
          const frame=queue.shift();
          setViseme(frame.vis);
          timer=setTimeout(stepQueue, frame.dur);
        }}

        function stopPlay(hard=false) {{
          playing = false; forceStop = hard || false; queue = [];
          if (timer) {{ clearTimeout(timer); timer = null; }}
          setViseme("rest");
          if (utterance) {{ speechSynthesis.cancel(); utterance = null; }}
        }}

        function playText(text) {{
          const t = (text || "").trim();
          if (!t) return;
          stopPlay();
          utterance = new SpeechSynthesisUtterance(t);
          if (selectedVoice) utterance.voice = selectedVoice;
          utterance.rate = 1.0;
          utterance.pitch = (currentCharacter === 'girl') ? 1.2 : 0.9;
          utterance.volume = 1.0;

          utterance.onstart = () => {{
            queue = prepareQueue(t);
            if (queue.length > 0) {{ playing = true; forceStop = false; stepQueue(); }}
          }};
          utterance.onend = () => stopPlay(true);
          utterance.onerror = () => stopPlay(true);

          speechSynthesis.speak(utterance);
        }}

        // Injected text from Streamlit:
        const INJECTED_TEXT = `{safe_text}`;

        // Set textarea (hidden) and auto-play on load
        document.addEventListener('DOMContentLoaded', () => {{
          textInput.value = INJECTED_TEXT;
          // default character: boy (image already set). You can switch by setting currentCharacter='girl' and image:
          // characterImage.src = characters['girl'].image; currentCharacterName.textContent = characters['girl'].name;
          setViseme("rest");
          // Give voices a moment to load on some browsers
          setTimeout(() => playText(INJECTED_TEXT), 200);
        }});
      </script>
    </body>
    </html>
    """
    # Render in sidebar (top)
    with st.sidebar:
        components.html(html, height=330, scrolling=False)


# ── Streamlit Page Configuration & State Initialization ──────────────────
st.set_page_config(page_title="Interactive Educational Agent", page_icon="🤖")

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
    st.title("🧑‍🎓 Interactive Simulation Educational Agent")
    st.info(f"Welcome! Ready to learn about **{concept_pkg.title}**? Click 'Start Learning' to begin your personalized learning session.")
    
    if st.button("🚀 Start Learning", type="primary"):
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
    with st.spinner("🤔 Thinking..."):
        try:
            # Check if this is the initial start
            if not st.session_state.messages:
                # Start the conversation
                agent_reply = st.session_state.agent.start()
            else:
                # Continue the conversation with the last user message
                last_user_msg = None
                for role, msg in reversed(st.session_state.messages):
                    if role == "user":
                        last_user_msg = msg
                        break
                
                if last_user_msg:
                    agent_reply = st.session_state.agent.post(last_user_msg)
                else:
                    agent_reply = "I'm waiting for your response."
            
            if agent_reply:
                st.session_state.messages.append(("assistant", agent_reply))
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            agent_reply = "I encountered an error. Please try again."
            st.session_state.messages.append(("assistant", agent_reply))
            st.stop()
    
    st.rerun()

# ── Main Application Logic & UI Display ──────────────────────────────────
st.title("🧑‍🎓 Interactive Simulation Educational Agent")

# Display session info in sidebar
with st.sidebar:
    # 1) VISeme at the very top (latest assistant message)
    last_assistant_text = None
    for role, msg in reversed(st.session_state.get("messages", [])):
        if role == "assistant" and isinstance(msg, str) and msg.strip():
            last_assistant_text = msg
            break

    # Render the character (auto-plays on each new assistant msg)
    render_viseme_sidebar(last_assistant_text or "Hello! I'm ready to explain 😊", key="viseme_iframe_top")

    # 2) The rest of your existing sidebar content
    st.header("📊 Session Info")
    if "agent" in st.session_state:
        session_info = st.session_state.agent.session_info()
        st.write(f"**Session ID:** {session_info['session_id']}")
        st.write(f"**User ID:** {session_info['user_id']}")
        st.write(f"**Current State:** {st.session_state.agent.current_state()}")
        st.write(f"**Concept:** {concept_pkg.title}")
        if session_info.get('tags'):
            st.write(f"**Tags:** {session_info['tags']}")

    st.markdown("---")
    st.markdown("**💡 How to interact:**")
    st.markdown("- Type your responses in the chat input")
    st.markdown("- Or use the microphone to speak")
    st.markdown("- The agent will guide you through learning")


# Display all messages. The audio player is only added for the last assistant message.
for i, (role, msg) in enumerate(st.session_state.messages):
    with st.chat_message(role):
        st.write(msg)
        
        # Check if we need to show simulation after this assistant message
        if role == "assistant" and (i == len(st.session_state.messages) - 1):
            # Display simulation if needed
            display_simulation_if_needed()
            
            # Add audio playback for the latest assistant message
            # try:
            #     play_text_as_audio(msg, st.container())
            # except Exception as e:
            #     st.caption("⚠️ Audio playback unavailable")
            #     st.stop()
            pass

# Handle user input at the bottom of the page
if "agent" in st.session_state and st.session_state.agent.current_state() != "END":
    user_msg = None
    
    # Audio input
    col1, col2 = st.columns([3, 1])
    with col2:
        st.caption("🎤 Voice Input")
        recorded_audio_bytes = audio_recorder(
            text="Click to speak",
            key=f"audio_recorder_{st.session_state.audio_recorder_key_counter}",
            icon_size="1x", 
            pause_threshold=2.0
        )
        
    if recorded_audio_bytes:
        with st.spinner("🎯 Transcribing..."):
            user_msg = transcribe_recorded_audio_bytes(recorded_audio_bytes)
            if user_msg and not user_msg.startswith("["):  # Valid transcription
                st.success(f"You said: {user_msg}")

    # Text input
    text_input = st.chat_input("💬 Type your response here...")
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
    st.success("🎉 Learning Session Complete!")
    
    # Get session summary from agent state
    session_summary = st.session_state.agent.state.get("session_summary", {})
    
    if session_summary:
        st.subheader("📋 Session Summary")
        st.json(session_summary)
        
        # Download session summary
        summary_json = json.dumps(session_summary, indent=2)
        st.download_button(
            label="📥 Download Session Summary", 
            data=summary_json, 
            file_name=f"session_summary_{st.session_state.session_id}.json", 
            mime="application/json"
        )
    else:
        st.info("Session completed successfully!")
    
    # Show session info for Langfuse tracking
    if "agent" in st.session_state:
        session_info = st.session_state.agent.session_info()
        st.subheader("🔍 Langfuse Session Details")
        st.code(f"Session ID: {session_info['session_id']}\nThread ID: {session_info['thread_id']}")
    
    # Compute and upload session metrics
    if "session_metrics_computed" not in st.session_state:
        with st.spinner("📊 Computing session metrics..."):
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
                st.success("✅ Session metrics computed and uploaded to Langfuse!")
            except Exception as e:
                st.error(f"❌ Failed to compute metrics: {e}")
                st.session_state.session_metrics_computed = True  # Mark as attempted to avoid retry
    
    # Display computed metrics
    if "session_metrics" in st.session_state:
        st.subheader("📊 Session Metrics")
        metrics = st.session_state.session_metrics
        
        # Key metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Quiz Score", f"{metrics.quiz_score:.1f}%")
            st.metric("User Type", metrics.user_type)
        with col2:
            st.metric("Engagement Rating", f"{metrics.user_engagement_rating:.1f}/5")
            st.metric("Interest Rating", f"{metrics.user_interest_rating:.1f}/5")
        with col3:
            st.metric("Concepts Covered", metrics.num_concepts_covered)
            st.metric("Enjoyment Probability", f"{metrics.enjoyment_probability:.0%}")
        
        # Download metrics
        metrics_json = metrics.model_dump_json(indent=2)
        st.download_button(
            label="📊 Download Session Metrics",
            data=metrics_json,
            file_name=f"session_metrics_{st.session_state.session_id}.json",
            mime="application/json"
        )
    
    # Option to start a new session
    if st.button("🔄 Start New Session", type="primary"):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Footer
st.markdown("---")
st.caption("🤖 Powered by Educational AI Agent | 📊 Tracked with Langfuse")
