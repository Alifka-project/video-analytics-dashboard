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

def test_camera(camera_index):
    """Test if camera is available"""
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        return ret and frame is not None
    return False

def main():
    st.title("📹 Video Analytics Dashboard")
    st.markdown("---")
    
    # Initialize state
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'camera_initialized' not in st.session_state:
        st.session_state.camera_initialized = False
    
    # Sidebar
    st.sidebar.header("⚙️ Controls")
    
    # Camera diagnostics
    st.sidebar.subheader("📷 Camera Setup")
    camera_index = st.sidebar.selectbox("Camera Index", [0, 1, 2], index=0)
    
    # Test camera button
    if st.sidebar.button("🔍 Test Camera"):
        with st.sidebar:
            with st.spinner("Testing camera..."):
                if test_camera(camera_index):
                    st.success(f"✅ Camera {camera_index} works!")
                else:
                    st.error(f"❌ Camera {camera_index} not available")
    
    # Control buttons
    st.sidebar.subheader("🎮 Control")
    
    col_start, col_stop = st.sidebar.columns(2)
    
    with col_start:
        if st.button("▶️ START", type="primary", use_container_width=True):
            st.session_state.running = True
            st.session_state.camera_initialized = False
    
    with col_stop:
        if st.button("⏹️ STOP", use_container_width=True):
            st.session_state.running = False
            st.session_state.camera_initialized = False
    
    # Analytics options
    st.sidebar.subheader("📊 Features")
    face_on = st.sidebar.checkbox("Face Detection", True)
    posture_on = st.sidebar.checkbox("Posture Analysis", True)
    gaze_on = st.sidebar.checkbox("Gaze Tracking", True)
    
    # Status display
    if st.session_state.running:
        st.sidebar.success("🟢 RUNNING")
    else:
        st.sidebar.info("🔴 STOPPED")
    
    # Troubleshooting
    with st.sidebar.expander("🛠️ Troubleshooting"):
        st.markdown("""
        **Camera not working?**
        1. Close other camera apps (Zoom, Teams, etc.)
        2. Try different camera index (0, 1, 2)
        3. Grant camera permissions in browser
        4. Restart browser if needed
        
        **On Mac:** Try Camera Index 1
        **On Windows:** Try opening Camera app first
        """)
    
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
    
    # Main camera logic
    if st.session_state.running:
        # Initialize camera if not done
        if not st.session_state.camera_initialized:
            with st.spinner("Initializing camera..."):
                # Test camera first
                if not test_camera(camera_index):
                    st.error(f"❌ Cannot access camera {camera_index}")
                    st.error("**Try these solutions:**")
                    st.error("• Close other apps using camera")
                    st.error("• Try different camera index")
                    st.error("• Check browser permissions")
                    st.session_state.running = False
                    return
                
                st.session_state.camera_initialized = True
                st.success("✅ Camera initialized!")
        
        # Capture frame
        cap = cv2.VideoCapture(camera_index)
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
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
                
                # Add timestamp and status
                timestamp = datetime.now().strftime("%H:%M:%S")
                cv2.putText(frame, f"LIVE | {timestamp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "STREAMING", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                video_area.image(frame, channels="BGR", use_column_width=True)
                
                # Update analytics
                face_status = "✅ Visible" if face_detected else "❌ Not Found"
                posture_status = "✅ Centered" if posture_centered else "⚠️ Off-Center"
                gaze_status = "✅ Looking" if looking_at_camera else "❌ Away"
                
                face_display.metric("👤 Face", face_status)
                posture_display.metric("🧍 Posture", posture_status)
                gaze_display.metric("👁️ Gaze", gaze_status)
                
                # Status summary
                status_html = f"""
                <div style="padding: 15px; border-radius: 10px; background-color: #d4edda;">
                    <h4>🟢 LIVE STREAMING</h4>
                    <p><strong>Face:</strong> {face_status}</p>
                    <p><strong>Posture:</strong> {posture_status}</p>
                    <p><strong>Gaze:</strong> {gaze_status}</p>
                    <p><strong>Camera:</strong> Working properly</p>
                </div>
                """
                status_display.markdown(status_html, unsafe_allow_html=True)
                
            else:
                video_area.error("❌ Cannot read from camera - try different camera index")
                st.session_state.running = False
        else:
            video_area.error("❌ Cannot open camera - check if it's being used by another app")
            st.session_state.running = False
        
        # Release camera
        cap.release()
        
        # Continue streaming immediately
        st.rerun()
    
    else:
        # Stopped state
        video_area.info("📷 Camera is OFF. Click ▶️ START to begin streaming.")
        
        face_display.metric("👤 Face", "⏸️ Ready")
        posture_display.metric("🧍 Posture", "⏸️ Ready")
        gaze_display.metric("👁️ Gaze", "⏸️ Ready")
        
        status_display.info("🔴 Stopped. Ready to start streaming.")

if __name__ == "__main__":
    main()
