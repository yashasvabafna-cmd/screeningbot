import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import tempfile
import os
import time
from transcriber import transcribe_audio
from agent import process_with_agent, start_interview
from tts import text_to_speech
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
import base64
import streamlit.components.v1 as components

def get_robo_html(audio_path, autoplay=True):
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    
    autoplay_attr = "autoplay" if autoplay else ""
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        * {{margin:0; padding:0; box-sizing:border-box;}}
        body {{font-family:'Inter',sans-serif; display:flex; justify-content:center; align-items:center; height:100%; background:transparent; overflow: hidden;}}
        .container {{background:rgba(255,255,255,0.95); backdrop-filter:blur(10px); border-radius:20px; padding:0.5rem; box-shadow:0 10px 30px rgba(0,0,0,0.1); text-align:center; width:100%;}}
        .anim-box {{margin:0.5rem 0; position:relative; display:flex; justify-content:center; align-items:center; height:180px;}}
        .controls {{margin-top:0.5rem;}}
        .play-btn {{background:linear-gradient(135deg,#1e3c72 0%,#2a5298 100%); color:white; border:none; padding:0.5rem 1.5rem; font-size:0.9rem; font-weight:600; border-radius:50px; cursor:pointer; transition:all 0.3s; box-shadow:0 5px 15px rgba(30,60,114,0.4); font-family:'Inter',sans-serif;}}
        .play-btn:hover {{transform:translateY(-2px); box-shadow:0 8px 20px rgba(30,60,114,0.6);}}
        .play-btn.playing {{background:linear-gradient(135deg,#4facfe 0%,#00f2fe 100%);}}
        .status {{margin-top:0.2rem; color:#666; font-size:0.7rem; font-weight:500;}}
        .progress-container {{width:100%; height:4px; background:#e0e0e0; border-radius:10px; margin-top:0.5rem; overflow:hidden;}}
        .progress-bar {{height:100%; background:linear-gradient(90deg,#1e3c72 0%,#2a5298 100%); width:0%; transition:width 0.1s linear; border-radius:10px;}}
        
        /* Happy Waving Robot Animation */
        .robo-container {{position:relative; width:200px; height:220px; display:flex; justify-content:center; transform: scale(0.65); transform-origin: center top;}}
        
        /* Head */
        .robo-head {{
            width: 180px; height: 160px;
            background: #dbe7f0; /* Bluish grey */
            border: 4px solid #1a2530;
            border-radius: 90px 90px 70px 70px;
            position: absolute; top: 40px; z-index: 10;
            box-shadow: inset -10px -10px 20px rgba(0,0,0,0.05);
        }}
        
        /* Face Panel */
        .face-panel {{
            width: 140px; height: 90px;
            background: #e1f5fe; /* Light blue */
            border: 3px solid #1a2530;
            border-radius: 40px 40px 30px 30px;
            position: absolute; top: 45px; left: 50%;
            transform: translateX(-50%);
            overflow: hidden;
        }}
        
        /* Forehead Panel */
        .forehead-panel {{
            width: 50px; height: 25px;
            background: #dbe7f0;
            border: 3px solid #1a2530;
            border-radius: 5px 5px 0 0;
            position: absolute; top: 20px; left: 50%;
            transform: translateX(-50%);
            z-index: 11;
        }}
        
        /* Ears/Headphones */
        .ear {{
            width: 25px; height: 60px;
            background: #f0f7fa;
            border: 3px solid #1a2530;
            border-radius: 15px;
            position: absolute; top: 90px;
            z-index: 5;
        }}
        .ear.left {{left: 45px;}}
        .ear.right {{right: 45px;}}
        
        /* Eyes (Visible & Blinking) */
        .eye {{
            width: 18px; height: 25px;
            background: #1a2530;
            border-radius: 50%;
            position: absolute; top: 25px;
            animation: blink 4s infinite;
        }}
        .eye.left {{left: 35px;}}
        .eye.right {{right: 35px;}}
        
        .eye::after {{
            content: '';
            width: 6px; height: 6px;
            background: white;
            border-radius: 50%;
            position: absolute; top: 5px; left: 4px;
        }}
        
        @keyframes blink {{
            0%, 48%, 52%, 100% {{transform: scaleY(1);}}
            50% {{transform: scaleY(0.1);}}
        }}
        
        /* Cheeks */
        .cheek {{
            width: 20px; height: 10px;
            background: #ff7e7e;
            border-radius: 50%;
            position: absolute; top: 55px;
            opacity: 0.6;
        }}
        .cheek.left {{left: 25px;}}
        .cheek.right {{right: 25px;}}
        
        /* Mouth (Realistic Speaking) */
        .mouth {{
            width: 30px; height: 15px;
            background: #1a2530;
            border-radius: 0 0 30px 30px;
            position: absolute; top: 60px; left: 50%;
            transform: translateX(-50%);
            transition: all 0.1s;
            overflow: hidden;
        }}
        .mouth::after {{
            content: '';
            width: 20px; height: 8px;
            background: #ff7e7e;
            border-radius: 50%;
            position: absolute; bottom: -4px; left: 50%;
            transform: translateX(-50%);
        }}
        
        .mouth.speaking {{
            animation: speak-real 0.2s infinite alternate;
        }}
        
        @keyframes speak-real {{
            0% {{height: 10px; border-radius: 5px;}}
            100% {{height: 25px; border-radius: 0 0 20px 20px;}}
        }}
        
        /* Body */
        .body {{
            width: 120px; height: 100px;
            background: #dbe7f0;
            border: 4px solid #1a2530;
            border-radius: 30px 30px 50px 50px;
            position: absolute; top: 190px; left: 50%;
            transform: translateX(-50%);
            z-index: 9;
        }}
        
        /* Chest Panel */
        .chest-panel {{
            width: 70px; height: 50px;
            background: #c3d0d9;
            border: 3px solid #1a2530;
            border-radius: 15px;
            position: absolute; top: 25px; left: 50%;
            transform: translateX(-50%);
        }}
        
        /* Neck */
        .neck {{
            width: 60px; height: 20px;
            background: #1a2530;
            position: absolute; top: 180px; left: 50%;
            transform: translateX(-50%);
            z-index: 8;
        }}
        
        /* Arms */
        .arm {{
            width: 25px; height: 70px;
            background: #dbe7f0;
            border: 3px solid #1a2530;
            border-radius: 15px;
            position: absolute; top: 200px;
            z-index: 8;
        }}
        .arm.left {{
            left: 70px;
            transform-origin: top center;
            transform: rotate(20deg);
        }}
        .arm.right {{
            right: 70px;
            transform: rotate(-20deg);
        }}
        
        /* Waving Hand */
        .hand-container {{
            position: absolute;
            top: 140px; left: 30px;
            z-index: 20;
            transform-origin: bottom right;
            animation: wave 2s ease-in-out infinite;
        }}
        @keyframes wave {{
            0%, 100% {{transform: rotate(0deg);}}
            50% {{transform: rotate(-20deg);}}
        }}
        
        .hand {{
            width: 40px; height: 50px;
            background: #dbe7f0;
            border: 3px solid #1a2530;
            border-radius: 15px 15px 10px 10px;
            position: relative;
        }}
        .finger {{
            width: 8px; height: 20px;
            background: #dbe7f0;
            border: 3px solid #1a2530;
            border-radius: 5px;
            position: absolute; top: -15px;
        }}
        .f1 {{left: -2px; transform: rotate(-20deg);}}
        .f2 {{left: 10px; top: -20px;}}
        .f3 {{left: 22px; top: -18px; transform: rotate(10deg);}}
        .thumb {{
            width: 10px; height: 25px;
            background: #dbe7f0;
            border: 3px solid #1a2530;
            border-radius: 5px;
            position: absolute; right: -8px; top: 15px;
            transform: rotate(40deg);
        }}
        .palm-detail {{
            width: 15px; height: 15px;
            background: #1a2530;
            border-radius: 50%;
            position: absolute; top: 20px; left: 10px;
            opacity: 0.2;
        }}
        
    </style>
    </head>
    <body>
    <div class="container">
        <div class="anim-box">
            <div class="robo-container">
                <!-- Ears (behind head) -->
                <div class="ear left"></div>
                <div class="ear right"></div>
                
                <!-- Head -->
                <div class="robo-head">
                    <div class="forehead-panel"></div>
                    <div class="face-panel">
                        <div class="eye left"></div>
                        <div class="eye right"></div>
                        <div class="cheek left"></div>
                        <div class="cheek right"></div>
                        <div class="mouth" id="roboMouth"></div>
                    </div>
                </div>
                
                <!-- Neck & Body -->
                <div class="neck"></div>
                <div class="body">
                    <div class="chest-panel"></div>
                </div>
                
                <!-- Arms -->
                <div class="hand-container">
                    <div class="hand">
                        <div class="finger f1"></div>
                        <div class="finger f2"></div>
                        <div class="finger f3"></div>
                        <div class="thumb"></div>
                        <div class="palm-detail"></div>
                    </div>
                </div>
                <div class="arm right"></div>
                
            </div>
        </div>
        
        <div class="controls">
            <button class="play-btn" id="playBtn">▶ Play Audio</button>
            <div class="status" id="status">Ready to play</div>
            <div class="progress-container"><div class="progress-bar" id="progressBar"></div></div>
        </div>
        
        <audio id="audioPlayer" preload="auto" {autoplay_attr}><source src="data:audio/wav;base64,{audio_base64}" type="audio/wav"></audio>
    </div>
    
    <script>
    const audio = document.getElementById('audioPlayer');
    const playBtn = document.getElementById('playBtn');
    const status = document.getElementById('status');
    const progressBar = document.getElementById('progressBar');
    let isPlaying = false;

    // Try to autoplay on load if attribute is present
    window.addEventListener('load', function() {{
        if (audio.hasAttribute('autoplay')) {{
            var playPromise = audio.play();
            if (playPromise !== undefined) {{
                playPromise.then(_ => {{
                    // Autoplay started!
                    console.log("Autoplay started successfully");
                }})
                .catch(error => {{
                    // Auto-play was prevented
                    console.log("Autoplay prevented:", error);
                    status.textContent = "Click Play to start";
                }});
            }}
        }}
    }});

    playBtn.addEventListener('click', function() {{
        if(isPlaying) {{
            audio.pause();
        }} else {{
            audio.play();
        }}
    }});

    audio.addEventListener('play', function() {{
        isPlaying = true;
        playBtn.textContent = '⏸ Pause';
        playBtn.classList.add('playing');
        status.textContent = 'Playing...';
        
        // Activate Robotic Face
        document.getElementById('roboMouth').classList.add('speaking');
    }});

    audio.addEventListener('pause', function() {{
        isPlaying = false;
        playBtn.textContent = '▶ Play Audio';
        playBtn.classList.remove('playing');
        status.textContent = 'Paused';
        
        // Deactivate Robotic Face
        document.getElementById('roboMouth').classList.remove('speaking');
    }});

    audio.addEventListener('ended', function() {{
        isPlaying = false;
        playBtn.textContent = '▶ Play Again';
        playBtn.classList.remove('playing');
        status.textContent = 'Finished';
        progressBar.style.width = '0%';
        
        // Deactivate Robotic Face
        document.getElementById('roboMouth').classList.remove('speaking');
    }});

    audio.addEventListener('timeupdate', function() {{
        const progress = (audio.currentTime / audio.duration) * 100;
        progressBar.style.width = progress + '%';
    }});
    </script>
    </body>
    </html>
    """

def response_generator(input_msg): 
    for word in input_msg.split(): 
        yield word + " " 
        time.sleep(0.01)  # Faster streaming for better UX

st.set_page_config(page_title="Sigmoid AI Screening", page_icon="🎙️", layout="wide")

st.title("🎙️ Sigmoid AI Candidate Screening")

# Custom CSS for better styling
st.markdown("""
<style>
    /* Hide audio player */
    .stAudio {
        display: none;
    }
    
    /* Improved styling */
    .main {
        background-color: #f8f9fa;
    }
    
    h1 {
        color: #1f2937;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    h2, h3 {
        color: #374151;
        font-weight: 600;
    }
    
    .stButton > button {
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Recording indicator animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .recording-indicator {
        animation: pulse 1.5s ease-in-out infinite;
        color: #ef4444;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.5rem;
        text-align: center;
    }
    
    /* Processing indicator */
    .processing-indicator {
        color: #3b82f6;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.5rem;
        text-align: center;
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Fixed Sidebar Width */
    section[data-testid="stSidebar"] {
        width: 280px !important;
        min-width: 280px !important;
        max-width: 280px !important;
    }
    
    /* Desktop Layout: Fixed Right Column */
    @media (min-width: 992px) {
        /* Target the Main Layout Horizontal Block */
        div[data-testid="stHorizontalBlock"]:nth-of-type(1) {
            align-items: flex-start;
        }

        /* Chat Column (Left) - Force width to avoid expansion */
        div[data-testid="stHorizontalBlock"]:nth-of-type(1) > div[data-testid="column"]:nth-of-type(1) {
            width: 65% !important;
            flex: none !important;
            min-width: 65% !important;
        }

        /* Robot Column (Right) - Fixed Position */
        div[data-testid="stHorizontalBlock"]:nth-of-type(1) > div[data-testid="column"]:nth-of-type(2) {
            position: fixed !important;
            top: 6rem;
            right: 2rem;
            width: 28% !important;
            z-index: 9999;
            height: auto !important;
            
            /* Panel Styling */
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 1rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.05);
            border: 1px solid rgba(255,255,255,0.5);
        }
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for interview configuration
with st.sidebar:
    st.subheader("📋 Interview Details")
    
    st.markdown("**Candidate:** AVIJIT SWAIN")
    st.markdown("**Job Role:** Data Scientist")
    st.markdown("**Interview Type:** HR Screening")
    st.markdown(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
    
    # Actions section
    if st.session_state.get('conversation_history', []):
        st.subheader("Actions")
        
        # Clear Conversation
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.displayed_messages = 0
            st.rerun()
            
        st.divider()
        
        st.subheader("📥 Export")
        
        def generate_pdf():
            """Generate PDF transcript"""
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter,
                                  rightMargin=72, leftMargin=72,
                                  topMargin=72, bottomMargin=18)
            
            # Container for the 'Flowable' objects
            elements = []
            
            # Define styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor='#1f2937',
                spaceAfter=30,
                alignment=TA_CENTER
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                textColor='#374151',
                spaceAfter=12,
                spaceBefore=12
            )
            
            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontSize=11,
                textColor='#1f2937',
                spaceAfter=8
            )
            
            role_style = ParagraphStyle(
                'RoleStyle',
                parent=styles['Normal'],
                fontSize=12,
                textColor='#0284c7',
                fontName='Helvetica-Bold',
                spaceAfter=4
            )
            
            # Add title
            elements.append(Paragraph("SIGMOID AI CANDIDATE SCREENING", title_style))
            elements.append(Spacer(1, 0.2*inch))
            
            # Add interview details
            elements.append(Paragraph("Interview Details", heading_style))
            elements.append(Paragraph(f"<b>Job Role:</b> Data Scientist", normal_style))
            elements.append(Paragraph(f"<b>Interview Type:</b> HR Screening", normal_style))
            elements.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
            elements.append(Spacer(1, 0.3*inch))
            
            # Add conversation
            elements.append(Paragraph("Conversation Transcript", heading_style))
            elements.append(Spacer(1, 0.2*inch))
            
            for i, msg in enumerate(st.session_state.conversation_history, 1):
                role = "CANDIDATE" if msg["role"] == "user" else "INTERVIEWER"
                elements.append(Paragraph(f"{role}:", role_style))
                # Escape special characters and preserve formatting
                content = msg['content'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                elements.append(Paragraph(content, normal_style))
                elements.append(Spacer(1, 0.15*inch))
            
            # Build PDF
            doc.build(elements)
            buffer.seek(0)
            return buffer
        
        # Download as PDF
        pdf_buffer = generate_pdf()
        st.download_button(
            label="📄 Download as PDF",
            data=pdf_buffer,
            file_name=f"interview_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# Define a class to handle recording state and data
class Recorder:
    def __init__(self):
        self.frames = []
        self.stream = None
        self.is_recording = False

    def start(self, fs):
        self.frames = []
        self.is_recording = True
        self.stream = sd.InputStream(
            samplerate=fs,
            channels=1,
            dtype='int16',
            callback=self._callback
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_recording = False

    def reset(self):
        self.stop()
        self.frames = []
        self.is_recording = False

    def _callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.frames.append(indata.copy())

    def get_audio(self):
        if not self.frames:
            return None
        return np.concatenate(self.frames, axis=0)

# Initialize session state
if 'recorder' not in st.session_state:
    st.session_state.recorder = Recorder()
if 'fs' not in st.session_state:
    st.session_state.fs = 44100
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'displayed_messages' not in st.session_state:
    st.session_state.displayed_messages = 0
if 'audio_processed' not in st.session_state:
    st.session_state.audio_processed = False
if 'interview_started' not in st.session_state:
    st.session_state.interview_started = False
if 'interview_active' not in st.session_state:
    st.session_state.interview_active = True
if 'latest_audio_path' not in st.session_state:
    st.session_state.latest_audio_path = None
if 'last_played_audio' not in st.session_state:
    st.session_state.last_played_audio = None

# Display conversation history
# st.subheader("💬 Conversation")

# Main Layout: Chat (Left) and Robot (Right)
col_chat, col_robot = st.columns([0.7, 0.3], gap="medium")

with col_robot:
    # Robot Player Section (Sticky)
    if st.session_state.get('latest_audio_path'):
        # Determine if we should autoplay
        # Autoplay if: 1) This is a new audio (different from last played), OR 2) This is the first audio (last_played_audio is None)
        should_autoplay = False
        current_audio = st.session_state.get('latest_audio_path')
        last_audio = st.session_state.get('last_played_audio')
        
        if current_audio != last_audio or last_audio is None:
            should_autoplay = True
            st.session_state.last_played_audio = current_audio
            
        robo_html = get_robo_html(st.session_state.latest_audio_path, autoplay=should_autoplay)
        components.html(robo_html, height=400)
    else:
        # Placeholder or default state if needed
        st.markdown("""
            <div style="text-align: center; padding: 2rem; color: #6b7280; background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🤖</div>
                <p>I'm ready to interview you!</p>
            </div>
        """, unsafe_allow_html=True)

with col_chat:
    # Start Interview Button
    if not st.session_state.interview_started:
        st.info("👋 Welcome! Click 'Start Interview' to begin the screening process.")
        
        if st.button("🎬 Start Interview", use_container_width=True, type="primary"):
            with st.spinner("Starting interview..."):
                # Call start_interview from agent
                agent_response, control_decision = start_interview()
                
                # Generate TTS for the greeting
                tts_audio_path = text_to_speech(agent_response)
                
                # Add assistant message to history
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": agent_response,
                    "audio_path": tts_audio_path
                })
                
                # Set latest audio for the robot
                st.session_state.latest_audio_path = tts_audio_path
                
                # Update interview state
                st.session_state.interview_started = True
                if control_decision == "stop":
                    st.session_state.interview_active = False
                
                st.rerun()
    else:
        if not st.session_state.conversation_history:
            st.info("👋 Interview started. Please use the recording buttons below.")

    for idx, message in enumerate(st.session_state.conversation_history):
        with st.chat_message(message["role"]):
            # Check if this is a new assistant message that needs streaming
            is_new_message = (message["role"] == "assistant" and 
                            idx >= st.session_state.displayed_messages)
            
            if is_new_message:
                # Stream the assistant's response manually (only for new messages)
                message_placeholder = st.empty()
                full_response = ""
                for word in message["content"].split():
                    full_response += word + " "
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.25)
                message_placeholder.markdown(message["content"])
            else:
                # Display already-shown messages normally (no streaming)
                st.markdown(message["content"])

    # Update displayed messages count
    st.session_state.displayed_messages = len(st.session_state.conversation_history)

    # Recording controls at the bottom
    st.divider()

    # Only show recording buttons if interview has started
    if st.session_state.interview_started:
        col1, col2 = st.columns(2)
        
        with col1:
            # Start button - disabled if interview is not active
            if st.button("🎤 Start Recording", disabled=st.session_state.recorder.is_recording or st.session_state.processing or not st.session_state.interview_active, use_container_width=True, type="primary"):
                try:
                    # Ensure any existing stream is closed first
                    if hasattr(st.session_state.recorder, 'stream') and st.session_state.recorder.stream:
                        st.session_state.recorder.stop()
                    
                    st.session_state.recorder.start(st.session_state.fs)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error starting recording: {e}")
                    st.session_state.recorder.is_recording = False
                    try:
                        st.session_state.recorder.stop()
                    except:
                        pass
        
        with col2:
            # Stop button
            if st.button("⏹️ Stop Recording", disabled=not st.session_state.recorder.is_recording, use_container_width=True, type="secondary"):
                st.session_state.recorder.stop()
                st.session_state.processing = True
                st.session_state.audio_processed = False  # Reset flag for new recording
                st.rerun()

    # Status indicators
    if st.session_state.recorder.is_recording:
        st.markdown('<div class="recording-indicator">🔴 Recording in progress... Speak clearly into your microphone</div>', unsafe_allow_html=True)

    if st.session_state.processing:
        st.markdown('<div class="processing-indicator">🤖 Just a moment...</div>', unsafe_allow_html=True)

    # Interview completion indicator
    if st.session_state.interview_started and not st.session_state.interview_active:
        st.success("✅ Interview completed! Thank you for your time.")

    # Process the recording
    if (st.session_state.processing and 
        not st.session_state.recorder.is_recording and 
        not st.session_state.audio_processed):
        
        audio_data = st.session_state.recorder.get_audio()
        
        if audio_data is not None:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Save user audio
                status_text.text("Processing user input...")
                progress_bar.progress(25)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as fp:
                    write(fp.name, st.session_state.fs, audio_data)
                    user_audio_path = fp.name
                
                # Transcribe
                transcription = transcribe_audio(audio_data, st.session_state.fs)
                progress_bar.progress(50)
                
                # Add user message to history immediately
                st.session_state.conversation_history.append({
                    "role": "user",
                    "content": transcription,
                    "audio_path": user_audio_path
                })
                
                # Mark audio as processed
                st.session_state.audio_processed = True
                
                # Clear progress indicators and rerun to show user message
                progress_bar.empty()
                status_text.empty()
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing: {e}")
                st.session_state.processing = False

    # Generate AI response if we just added a user message
    if (st.session_state.processing and 
        st.session_state.conversation_history and 
        st.session_state.conversation_history[-1]["role"] == "user"):
        
        # Show progress for AI generation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Get the last user message
            last_user_message = st.session_state.conversation_history[-1]["content"]
            
            # Get agent response
            status_text.text("Generating AI response...")
            progress_bar.progress(33)
            
            agent_response, control_decision = process_with_agent(last_user_message)
            
            # Generate TTS for agent response
            progress_bar.progress(66)
            
            tts_audio_path = text_to_speech(agent_response)
            
            progress_bar.progress(100)
            status_text.text("✅ Complete!")
            
            # Add assistant message to history
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": agent_response,
                "audio_path": tts_audio_path
            })
            
            # Set latest audio for the robot
            st.session_state.latest_audio_path = tts_audio_path
            
            # Check control decision and update interview state
            if control_decision == "stop":
                st.session_state.interview_active = False
            
        except Exception as e:
            st.error(f"Error generating AI response: {e}")
        
        # Reset processing flag
        st.session_state.processing = False
        progress_bar.empty()
        status_text.empty()
        st.rerun()
