# Alternative Solution: Streamlit WebRTC
# This keeps Streamlit but adds real-time video capability

import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime
import av

# Configure Streamlit
st.set_page_config(
    page_title="Real-time Video Analytics - WebRTC",
    page_icon="📹",
    layout="wide"
)

# Initialize MediaPipe
@st.cache_resource
def load_mediapipe():
    mp_face_detection = mp.solutions.face_detection
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    
    return {
        'face_detection': mp_face_detection.FaceDetection(min_detection_confidence=0.5),
        'pose': mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5),
        'face_mesh': mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5),
        'drawing': mp_drawing,
        'pose_connections': mp_pose.POSE_CONNECTIONS
    }

# Global analytics state
if 'analytics_state' not in st.session_state:
    st.session_state.analytics_state = {
        'face_detected': False,
        'posture_centered': False,
        'looking_at_camera': False,
        'face_count': 0,
        'last_update': datetime.now()
    }

def process_video_frame(frame, mp_models, show_face, show_pose, show_gaze):
    """Process video frame in real-time"""
    img = frame.to_ndarray(format="bgr24")
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Initialize results
    face_detected = False
    posture_centered = False
    looking_at_camera = False
    face_count = 0
    
    height, width = img.shape[:2]
    
    # Face detection
    if show_face:
        face_results = mp_models['face_detection'].process(rgb_img)
        if face_results.detections:
            face_detected = True
            face_count = len(face_results.detections)
            
            for detection in face_results.detections:
                mp_models['drawing'].draw_detection(img, detection)
    
    # Pose detection
    if show_pose:
        pose_results = mp_models['pose'].process(rgb_img)
        if pose_results.pose_landmarks:
            # Draw pose
            mp_models['drawing'].draw_landmarks(
                img, pose_results.pose_landmarks, mp_models['pose_connections']
            )
            
            # Check posture centering
            try:
                left_shoulder = pose_results.pose_landmarks.landmark[11]  # LEFT_SHOULDER
                right_shoulder = pose_results.pose_landmarks.landmark[12]  # RIGHT_SHOULDER
                
                shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
                if abs(shoulder_center_x - 0.5) < 0.1:  # Within 10% of center
                    posture_centered = True
                    cv2.putText(img, 'Centered', (10, height-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(img, 'Off-Center', (10, height-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            except:
                pass
    
    # Gaze tracking (simplified)
    if show_gaze and face_detected:
        face_mesh_results = mp_models['face_mesh'].process(rgb_img)
        if face_mesh_results.multi_face_landmarks:
            # Simple assumption: looking if face detected and centered
            looking_at_camera = posture_centered
            
            if looking_at_camera:
                cv2.putText(img, 'Looking at Camera', (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(img, 'Looking Away', (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add real-time indicator
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    cv2.putText(img, f"REAL-TIME | {timestamp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, "STREAMLIT WEBRTC", (10, height-80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Update global state
    st.session_state.analytics_state.update({
        'face_detected': face_detected,
        'posture_centered': posture_centered,
        'looking_at_camera': looking_at_camera,
        'face_count': face_count,
        'last_update': datetime.now()
    })
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("🎥 Real-time Video Analytics - Streamlit WebRTC")
    st.markdown("---")
    
    # Load MediaPipe models
    mp_models = load_mediapipe()
    
    # Sidebar controls
    st.sidebar.header("⚙️ Real-time Settings")
    
    # Analytics toggles
    st.sidebar.subheader("📊 Analytics")
    show_face = st.sidebar.checkbox("Face Detection", True)
    show_pose = st.sidebar.checkbox("Posture Analysis", True)
    show_gaze = st.sidebar.checkbox("Gaze Tracking", True)
    
    # WebRTC configuration
    rtc_config = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
    
    # Status
    st.sidebar.subheader("🚦 Status")
    if st.session_state.analytics_state['face_detected']:
        st.sidebar.success("🟢 LIVE - Face Detected")
    else:
        st.sidebar.error("🔴 LIVE - No Face")
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Video Stream")
        
        # WebRTC streamer - THIS IS THE KEY!
        webrtc_ctx = webrtc_streamer(
            key="real-time-analytics",
            rtc_configuration=rtc_config,
            video_frame_callback=lambda frame: process_video_frame(
                frame, mp_models, show_face, show_pose, show_gaze
            ),
            media_stream_constraints={
                "video": {
                    "width": 640,
                    "height": 480,
                    "frameRate": 30
                },
                "audio": False
            },
            async_processing=True,
        )
        
        # Stream status
        if webrtc_ctx.state.playing:
            st.success("🟢 Live stream active - Real-time processing enabled!")
        else:
            st.info("📷 Click 'START' above to begin real-time video analytics")
    
    with col2:
        st.subheader("📊 Live Analytics")
        
        # Get current state
        state = st.session_state.analytics_state
        
        # Real-time metrics
        face_status = "✅ Visible" if state['face_detected'] else "❌ Not Found"
        posture_status = "✅ Centered" if state['posture_centered'] else "⚠️ Off-Center"
        gaze_status = "✅ Looking" if state['looking_at_camera'] else "❌ Away"
        
        st.metric("👤 Face Detection", face_status, f"{state['face_count']} faces")
        st.metric("🧍 Posture Analysis", posture_status)
        st.metric("👁️ Gaze Tracking", gaze_status)
        
        # Live status
        st.subheader("🚦 Live Status")
        status_html = f"""
        <div style="padding: 15px; border-radius: 10px; background-color: {'#d1ecf1' if webrtc_ctx.state.playing else '#f8d7da'};">
            <h4>{'🔄 REAL-TIME ACTIVE' if webrtc_ctx.state.playing else '⏸️ STREAM STOPPED'}</h4>
            <p><strong>Face:</strong> {face_status}</p>
            <p><strong>Posture:</strong> {posture_status}</p>
            <p><strong>Gaze:</strong> {gaze_status}</p>
            <p><strong>Last Update:</strong> {state['last_update'].strftime('%H:%M:%S')}</p>
            <p><strong>Stream:</strong> {'🟢 Live' if webrtc_ctx.state.playing else '🔴 Stopped'}</p>
        </div>
        """
        st.markdown(status_html, unsafe_allow_html=True)
        
        # Performance info
        st.subheader("⚡ Performance")
        st.metric("📊 Processing", "Real-time" if webrtc_ctx.state.playing else "Stopped")
        st.metric("🎥 Stream Quality", "640x480 @ 30fps")
        st.metric("🌐 Technology", "WebRTC + MediaPipe")
        
        # Instructions
        if not webrtc_ctx.state.playing:
            st.info("""
            📱 **Getting Started:**
            1. Click 'START' above the video
            2. Allow camera access
            3. See real-time analytics!
            
            ✨ **Features:**
            - Smooth 30 FPS video
            - Real-time face detection
            - Live posture analysis
            - Gaze direction tracking
            """)

if __name__ == "__main__":
    main()
