import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="Simple Video Analytics",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded"
)

def detect_face_opencv(frame):
    """Detect faces using OpenCV's built-in cascade classifier"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, 'Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        return frame, len(faces) > 0
    except Exception as e:
        return frame, False

def analyze_posture_simple(frame):
    """Simple posture analysis based on frame composition"""
    try:
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        center_region = gray[height//3:2*height//3, width//3:2*width//3]
        center_intensity = np.mean(center_region)
        is_centered = center_intensity > 50
        
        cv2.rectangle(frame, (width//3, height//3), (2*width//3, 2*height//3), (0, 255, 0), 2)
        cv2.putText(frame, 'Center Zone', (width//3, height//3-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame, is_centered
    except Exception as e:
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
        else:
            cv2.putText(frame, 'Not Looking', (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame, looking_at_camera
    except Exception as e:
        return frame, False

def main():
    st.title("📹 Simple Video Analytics - Just Start/Stop")
    st.markdown("---")
    
    # Initialize simple state
    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Settings")
    
    # Camera settings
    camera_index = st.sidebar.selectbox("Camera Index", [0, 1, 2], index=0)
    
    # Analytics settings
    st.sidebar.subheader("📊 Analytics")
    show_face_detection = st.sidebar.checkbox("Face Detection", value=True)
    show_posture_analysis = st.sidebar.checkbox("Posture Analysis", value=True)
    show_gaze_tracking = st.sidebar.checkbox("Gaze Tracking", value=True)
    
    # Simple start/stop buttons
    st.sidebar.subheader("🎮 Camera Control")
    
    if st.sidebar.button("🚀 Start Camera", type="primary"):
        st.session_state.camera_on = True
    
    if st.sidebar.button("⏹️ Stop Camera"):
        st.session_state.camera_on = False
    
    # Status display
    if st.session_state.camera_on:
        st.sidebar.success("🟢 Camera: ON")
    else:
        st.sidebar.info("🔴 Camera: OFF")
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Video Feed")
        video_placeholder = st.empty()
    
    with col2:
        st.subheader("📊 Analytics")
        
        # Metrics placeholders
        face_metric = st.empty()
        posture_metric = st.empty()
        gaze_metric = st.empty()
        
        # Status indicators
        st.subheader("🚦 Status")
        status_placeholder = st.empty()
    
    # Simple camera processing - NO session management!
    if st.session_state.camera_on:
        try:
            # Open camera for single frame
            cap = cv2.VideoCapture(camera_index)
            
            if cap.isOpened():
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    # Flip frame for mirror effect
                    frame = cv2.flip(frame, 1)
                    
                    # Initialize analytics results
                    face_visible = False
                    posture_centered = False
                    looking_at_camera = False
                    
                    # Apply analytics
                    if show_face_detection:
                        frame, face_visible = detect_face_opencv(frame)
                    
                    if show_posture_analysis:
                        frame, posture_centered = analyze_posture_simple(frame)
                    
                    if show_gaze_tracking and face_visible:
                        frame, looking_at_camera = analyze_gaze_simple(frame, face_visible)
                    
                    # Add simple timestamp
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    cv2.putText(frame, f"Time: {timestamp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Display frame
                    video_placeholder.image(frame, channels="BGR", use_column_width=True)
                    
                    # Update metrics
                    face_status = "Visible" if face_visible else "Not Detected"
                    posture_status = "Centered" if posture_centered else "Off-Center"
                    gaze_status = "Looking" if looking_at_camera else "Looking Away"
                    
                    face_metric.metric("👤 Face", face_status)
                    posture_metric.metric("🧍 Posture", posture_status)
                    gaze_metric.metric("👁️ Gaze", gaze_status)
                    
                    # Simple status
                    status_html = f"""
                    <div style="padding: 15px; border-radius: 10px; background-color: #d4edda;">
                        <h4>📊 Current Status</h4>
                        <p><strong>Face:</strong> {face_status}</p>
                        <p><strong>Posture:</strong> {posture_status}</p>
                        <p><strong>Gaze:</strong> {gaze_status}</p>
                        <p><strong>Camera:</strong> 🟢 Active</p>
                    </div>
                    """
                    status_placeholder.markdown(status_html, unsafe_allow_html=True)
                    
                else:
                    video_placeholder.error("❌ Cannot read from camera")
                    st.session_state.camera_on = False
            else:
                video_placeholder.error("❌ Cannot open camera")
                st.session_state.camera_on = False
            
            # Always close camera immediately
            cap.release()
            
            # Auto-refresh only when camera is on
            time.sleep(0.1)  # Small delay
            st.rerun()
            
        except Exception as e:
            st.error(f"Camera error: {str(e)}")
            st.session_state.camera_on = False
    
    else:
        # Camera is off - show default state
        video_placeholder.info("📷 Camera is OFF - Click 'Start Camera' to begin")
        
        # Show ready state
        face_metric.metric("👤 Face", "Ready")
        posture_metric.metric("🧍 Posture", "Ready")
        gaze_metric.metric("👁️ Gaze", "Ready")
        
        status_html = """
        <div style="padding: 15px; border-radius: 10px; background-color: #f8f9fa;">
            <h4>📊 System Ready</h4>
            <p><strong>Camera:</strong> 🔴 OFF</p>
            <p><strong>Status:</strong> Waiting to start</p>
        </div>
        """
        status_placeholder.markdown(status_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
