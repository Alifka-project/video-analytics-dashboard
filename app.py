import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="Continuous Video Analytics",
    page_icon="📹",
    layout="wide"
)

def detect_face_opencv(frame):
    """Detect faces using OpenCV"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, 'Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        return frame, len(faces) > 0
    except:
        return frame, False

def analyze_posture_simple(frame):
    """Simple posture analysis"""
    try:
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        center_region = gray[height//3:2*height//3, width//3:2*width//3]
        center_intensity = np.mean(center_region)
        is_centered = center_intensity > 50
        
        cv2.rectangle(frame, (width//3, height//3), (2*width//3, 2*height//3), (0, 255, 0), 2)
        cv2.putText(frame, 'Center Zone', (width//3, height//3-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame, is_centered
    except:
        return frame, False

def analyze_gaze_simple(frame, face_detected):
    """Simple gaze analysis"""
    try:
        if not face_detected:
            return frame, False
        
        height, width = frame.shape[:2]
        looking_at_camera = face_detected
        
        if looking_at_camera:
            cv2.putText(frame, 'Looking at Camera', (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame, looking_at_camera
    except:
        return frame, False

def main():
    st.title("🎥 Continuous Video Analytics - Always Running")
    st.markdown("---")
    
    # Initialize ONLY if not exists
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = True  # Start automatically!
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    camera_index = st.sidebar.selectbox("Camera Index", [0, 1, 2], index=0)
    
    # Analytics toggles
    st.sidebar.subheader("📊 Analytics")
    face_on = st.sidebar.checkbox("Face Detection", True)
    posture_on = st.sidebar.checkbox("Posture Analysis", True)
    gaze_on = st.sidebar.checkbox("Gaze Tracking", True)
    
    # Simple toggle button
    if st.session_state.camera_active:
        if st.sidebar.button("⏸️ PAUSE", type="secondary"):
            st.session_state.camera_active = False
        st.sidebar.success("🟢 LIVE STREAMING")
    else:
        if st.sidebar.button("▶️ RESUME", type="primary"):
            st.session_state.camera_active = True
        st.sidebar.info("⏸️ PAUSED")
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    # Video area
    with col1:
        st.subheader("📹 Live Camera Feed")
        video_area = st.empty()
    
    # Analytics area
    with col2:
        st.subheader("📊 Real-time Analytics")
        face_display = st.empty()
        posture_display = st.empty()
        gaze_display = st.empty()
        
        st.subheader("📋 Live Status")
        status_display = st.empty()
    
    # CONTINUOUS CAMERA PROCESSING - NO STOPPING!
    if st.session_state.camera_active:
        # Capture frame
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if cap.isOpened():
            ret, frame = cap.read()
            
            if ret and frame is not None:
                # Mirror effect
                frame = cv2.flip(frame, 1)
                
                # Analytics
                face_detected = False
                posture_centered = False
                looking_at_camera = False
                
                if face_on:
                    frame, face_detected = detect_face_opencv(frame)
                
                if posture_on:
                    frame, posture_centered = analyze_posture_simple(frame)
                
                if gaze_on and face_detected:
                    frame, looking_at_camera = analyze_gaze_simple(frame, face_detected)
                
                # Add continuous indicator
                timestamp = datetime.now().strftime("%H:%M:%S")
                cv2.putText(frame, f"CONTINUOUS | {timestamp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, "ALWAYS RUNNING", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                video_area.image(frame, channels="BGR", use_column_width=True)
                
                # Update analytics
                face_status = "✅ Visible" if face_detected else "❌ Not Found"
                posture_status = "✅ Centered" if posture_centered else "⚠️ Off-Center"
                gaze_status = "✅ Looking" if looking_at_camera else "❌ Away"
                
                face_display.metric("👤 Face", face_status)
                posture_display.metric("🧍 Posture", posture_status)
                gaze_display.metric("👁️ Gaze", gaze_status)
                
                # Status display
                status_html = f"""
                <div style="padding: 15px; border-radius: 10px; background-color: #d1ecf1;">
                    <h4>🔄 CONTINUOUS MODE</h4>
                    <p><strong>Face:</strong> {face_status}</p>
                    <p><strong>Posture:</strong> {posture_status}</p>
                    <p><strong>Gaze:</strong> {gaze_status}</p>
                    <p><strong>Status:</strong> 🟢 Always Running</p>
                </div>
                """
                status_display.markdown(status_html, unsafe_allow_html=True)
                
            else:
                # Camera error - but keep trying!
                video_area.warning("🔄 Camera reconnecting...")
        
        # Release and immediately restart
        cap.release()
        
        # IMMEDIATE REFRESH - NO DELAYS, NO STOPS!
        st.rerun()
    
    else:
        # Paused state
        video_area.info("⏸️ Camera paused. Click RESUME to continue.")
        
        face_display.metric("👤 Face", "⏸️ Paused")
        posture_display.metric("🧍 Posture", "⏸️ Paused")
        gaze_display.metric("👁️ Gaze", "⏸️ Paused")
        
        status_display.info("⏸️ Paused - Click RESUME to restart continuous mode")

if __name__ == "__main__":
    main()
