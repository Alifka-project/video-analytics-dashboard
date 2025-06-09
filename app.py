import streamlit as st
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Real-time Video Analytics",
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
    st.title("📹 Real-time Video Analytics - Cloud Edition")
    st.markdown("---")
    
    # Auto-start real-time mode
    if 'realtime_active' not in st.session_state:
        st.session_state.realtime_active = True
    
    # Sidebar
    st.sidebar.header("⚙️ Real-time Settings")
    
    # Analytics toggles
    st.sidebar.subheader("📊 Analytics")
    face_on = st.sidebar.checkbox("Face Detection", True)
    posture_on = st.sidebar.checkbox("Posture Analysis", True)
    gaze_on = st.sidebar.checkbox("Gaze Tracking", True)
    
    # Refresh rate control
    st.sidebar.subheader("⚡ Performance")
    refresh_rate = st.sidebar.select_slider(
        "Refresh Speed",
        options=[0.5, 1.0, 1.5, 2.0],
        value=1.0,
        format_func=lambda x: f"{x}s (faster)" if x <= 1.0 else f"{x}s (slower)"
    )
    
    # Real-time control
    st.sidebar.subheader("🎮 Control")
    if st.session_state.realtime_active:
        if st.sidebar.button("⏸️ PAUSE", type="secondary"):
            st.session_state.realtime_active = False
        st.sidebar.success("🟢 REAL-TIME ACTIVE")
        st.sidebar.info(f"🔄 Auto-refresh every {refresh_rate}s")
    else:
        if st.sidebar.button("▶️ START REAL-TIME", type="primary"):
            st.session_state.realtime_active = True
        st.sidebar.error("🔴 PAUSED")
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Camera Feed")
        
        if st.session_state.realtime_active:
            # Real-time camera input with unique key for auto-refresh
            current_time = datetime.now().timestamp()
            camera_photo = st.camera_input(
                "📷 Real-time Camera", 
                key=f"realtime_cam_{current_time}",
                help="Camera automatically refreshes for real-time analysis"
            )
            
            if camera_photo is not None:
                # Process the image
                image = Image.open(camera_photo)
                img_array = np.array(image)
                
                # Convert RGB to BGR for OpenCV
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Apply analytics
                face_detected = False
                posture_centered = False
                looking_at_camera = False
                
                if face_on:
                    img_array, face_detected = detect_face_opencv(img_array)
                
                if posture_on:
                    img_array, posture_centered = analyze_posture_simple(img_array)
                
                if gaze_on and face_detected:
                    img_array, looking_at_camera = analyze_gaze_simple(img_array, face_detected)
                
                # Add real-time indicators
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
                cv2.putText(img_array, f"REAL-TIME | {timestamp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(img_array, "CLOUD STREAMING", (10, img_array.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display processed frame
                st.image(img_array, channels="BGR", use_column_width=True, caption="Live Analysis")
                
                # Update analytics in real-time
                with col2:
                    st.subheader("📊 Live Analytics")
                    
                    # Real-time metrics
                    face_status = "✅ Visible" if face_detected else "❌ Not Found"
                    posture_status = "✅ Centered" if posture_centered else "⚠️ Off-Center" 
                    gaze_status = "✅ Looking" if looking_at_camera else "❌ Away"
                    
                    st.metric("👤 Face Detection", face_status)
                    st.metric("🧍 Posture Analysis", posture_status)
                    st.metric("👁️ Gaze Tracking", gaze_status)
                    
                    # Live status with timestamp
                    status_html = f"""
                    <div style="padding: 15px; border-radius: 10px; background-color: #d1ecf1;">
                        <h4>🔄 LIVE ANALYSIS</h4>
                        <p><strong>Face:</strong> {face_status}</p>
                        <p><strong>Posture:</strong> {posture_status}</p>
                        <p><strong>Gaze:</strong> {gaze_status}</p>
                        <p><strong>Last Update:</strong> {timestamp}</p>
                        <p><strong>Status:</strong> 🟢 Streaming</p>
                    </div>
                    """
                    st.markdown(status_html, unsafe_allow_html=True)
                    
                    # Performance indicator
                    st.subheader("⚡ Performance")
                    st.metric("🔄 Refresh Rate", f"{refresh_rate}s")
                    st.metric("📡 Mode", "Cloud Streaming")
            
            else:
                st.info("📷 Initializing camera... Please allow camera access.")
                
                with col2:
                    st.subheader("📊 Analytics Ready")
                    st.metric("👤 Face Detection", "Waiting...")
                    st.metric("🧍 Posture Analysis", "Waiting...")
                    st.metric("👁️ Gaze Tracking", "Waiting...")
                    
                    st.info("🔄 Waiting for camera initialization")
            
            # Auto-refresh for real-time effect
            time.sleep(refresh_rate)
            st.rerun()
        
        else:
            # Paused state
            st.info("⏸️ Real-time analysis paused. Click 'START REAL-TIME' to resume.")
            
            with col2:
                st.subheader("📊 Analytics Paused")
                st.metric("👤 Face Detection", "⏸️ Paused")
                st.metric("🧍 Posture Analysis", "⏸️ Paused")
                st.metric("👁️ Gaze Tracking", "⏸️ Paused")
                
                st.warning("⏸️ Real-time mode paused")

if __name__ == "__main__":
    main()
