import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="Video Analytics",
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
    st.title("📹 Video Analytics Dashboard")
    st.markdown("---")
    
    # Simple state - just ON/OFF
    if 'running' not in st.session_state:
        st.session_state.running = False
    
    # Sidebar
    st.sidebar.header("⚙️ Controls")
    
    camera_index = st.sidebar.selectbox("Camera", [0, 1, 2], index=0)
    
    # Simple ON/OFF buttons
    col_start, col_stop = st.sidebar.columns(2)
    
    with col_start:
        if st.button("▶️ ON", type="primary"):
            st.session_state.running = True
    
    with col_stop:
        if st.button("⏹️ OFF"):
            st.session_state.running = False
    
    # Analytics options
    st.sidebar.subheader("📊 Features")
    face_on = st.sidebar.checkbox("Face Detection", True)
    posture_on = st.sidebar.checkbox("Posture Analysis", True)
    gaze_on = st.sidebar.checkbox("Gaze Tracking", True)
    
    # Show current status
    if st.session_state.running:
        st.sidebar.success("🟢 RUNNING")
    else:
        st.sidebar.info("🔴 STOPPED")
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    # Video area
    with col1:
        st.subheader("📹 Camera Feed")
        video_area = st.empty()
    
    # Analytics area
    with col2:
        st.subheader("📊 Analytics")
        face_display = st.empty()
        posture_display = st.empty()
        gaze_display = st.empty()
        
        st.subheader("📋 Status")
        status_display = st.empty()
    
    # Main camera logic - NO sessions, NO completion!
    if st.session_state.running:
        # Capture one frame
        cap = cv2.VideoCapture(camera_index)
        
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
                
                # Add timestamp
                timestamp = datetime.now().strftime("%H:%M:%S")
                cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display everything
                video_area.image(frame, channels="BGR", use_column_width=True)
                
                # Update metrics
                face_display.metric("👤 Face", "✅ Detected" if face_detected else "❌ Not Found")
                posture_display.metric("🧍 Posture", "✅ Centered" if posture_centered else "⚠️ Off-Center")
                gaze_display.metric("👁️ Gaze", "✅ Looking" if looking_at_camera else "❌ Away")
                
                # Status
                status_display.success("🟢 Camera Active - All systems working!")
            else:
                video_area.error("❌ Cannot read from camera")
        else:
            video_area.error("❌ Cannot open camera")
        
        # Close camera immediately
        cap.release()
        
        # Keep running - auto refresh
        time.sleep(0.2)
        st.rerun()
    
    else:
        # Stopped state
        video_area.info("📷 Camera is OFF. Click ▶️ ON to start.")
        
        face_display.metric("👤 Face", "⏸️ Paused")
        posture_display.metric("🧍 Posture", "⏸️ Paused")
        gaze_display.metric("👁️ Gaze", "⏸️ Paused")
        
        status_display.info("🔴 Camera stopped. Ready to start.")

if __name__ == "__main__":
    main()
