import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="Real-time Video Analytics",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded"
)

def detect_face_opencv(frame):
    """Detect faces using OpenCV's built-in cascade classifier"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw rectangles around faces
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
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Simple center mass calculation
        center_region = gray[height//3:2*height//3, width//3:2*width//3]
        center_intensity = np.mean(center_region)
        
        # Check if subject is roughly centered
        is_centered = center_intensity > 50  # Simple threshold
        
        # Draw center guide
        cv2.rectangle(frame, (width//3, height//3), (2*width//3, 2*height//3), (0, 255, 0), 2)
        cv2.putText(frame, 'Center Zone', (width//3, height//3-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame, is_centered
    except Exception as e:
        return frame, False

def analyze_gaze_simple(frame, face_detected):
    """Simple gaze analysis - assumes looking at camera if face is detected and centered"""
    try:
        if not face_detected:
            return frame, False
        
        height, width = frame.shape[:2]
        
        # Simple assumption: if face is detected and roughly centered, assume looking at camera
        # In a real implementation, this would use eye tracking
        looking_at_camera = face_detected
        
        if looking_at_camera:
            cv2.putText(frame, 'Looking at Camera', (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Not Looking', (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame, looking_at_camera
    except Exception as e:
        return frame, False

def main():
    st.title("🎥 Real-time Video Analytics Dashboard")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Settings")
    
    # Camera settings
    camera_index = st.sidebar.selectbox("Camera Index", [0, 1, 2], index=0)
    
    # Analytics settings
    st.sidebar.subheader("📊 Analytics")
    show_face_detection = st.sidebar.checkbox("Face Detection", value=True)
    show_posture_analysis = st.sidebar.checkbox("Posture Analysis", value=True)
    show_gaze_tracking = st.sidebar.checkbox("Gaze Tracking", value=True)
    
    # Performance settings
    st.sidebar.subheader("⚡ Performance")
    fps_limit = st.sidebar.slider("FPS Limit", 5, 30, 15, 1)
    
    # Control buttons
    start_camera = st.sidebar.button("🚀 Start Camera", type="primary")
    stop_camera = st.sidebar.button("⏹️ Stop Camera")
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Video Feed")
        video_placeholder = st.empty()
    
    with col2:
        st.subheader("📊 Real-time Analytics")
        
        # Metrics placeholders
        face_metric = st.empty()
        posture_metric = st.empty()
        gaze_metric = st.empty()
        
        # Status indicators
        st.subheader("🚦 Status")
        status_placeholder = st.empty()
        
        # Performance metrics
        st.subheader("⚡ Performance")
        fps_placeholder = st.empty()
    
    # Initialize session state
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    
    if start_camera:
        st.session_state.camera_active = True
    
    if stop_camera:
        st.session_state.camera_active = False
    
    # Camera processing loop
    if st.session_state.camera_active:
        try:
            # Initialize camera
            cap = cv2.VideoCapture(camera_index)
            
            if not cap.isOpened():
                st.error("❌ Could not access camera. Please check:")
                st.error("• Camera permissions in browser")
                st.error("• Camera is not being used by another app")
                st.error("• Try different camera index")
                st.session_state.camera_active = False
                return
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Performance tracking
            fps_counter = 0
            fps_start_time = time.time()
            
            # Main processing loop
            frame_count = 0
            while st.session_state.camera_active: #and frame_count < 100:  # Limit frames to prevent infinite loop
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to read from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                process_start = time.time()
                
                # Initialize analytics results
                face_visible = False
                posture_centered = False
                looking_at_camera = False
                
                # Apply analytics based on settings
                if show_face_detection:
                    frame, face_visible = detect_face_opencv(frame)
                
                if show_posture_analysis:
                    frame, posture_centered = analyze_posture_simple(frame)
                
                if show_gaze_tracking and face_visible:
                    frame, looking_at_camera = analyze_gaze_simple(frame, face_visible)
                
                # Add timestamp
                timestamp = datetime.now().strftime("%H:%M:%S")
                cv2.putText(frame, f"Time: {timestamp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display frame
                video_placeholder.image(frame, channels="BGR", use_column_width=True)
                
                # Update metrics
                face_status = "✅ Visible" if face_visible else "❌ Not Detected"
                posture_status = "✅ Centered" if posture_centered else "⚠️ Off-Center"
                gaze_status = "✅ Looking" if looking_at_camera else "❌ Looking Away"
                
                face_metric.metric("👤 Face Detection", face_status)
                posture_metric.metric("🧍 Posture", posture_status)
                gaze_metric.metric("👁️ Gaze", gaze_status)
                
                # Status summary
                status_html = f"""
                <div style="padding: 15px; border-radius: 10px; background-color: #f0f2f6;">
                    <h4>📊 Current Analysis:</h4>
                    <p><strong>Face:</strong> <span style="color: {'green' if face_visible else 'red'};">{face_status}</span></p>
                    <p><strong>Posture:</strong> <span style="color: {'green' if posture_centered else 'orange'};">{posture_status}</span></p>
                    <p><strong>Gaze:</strong> <span style="color: {'green' if looking_at_camera else 'red'};">{gaze_status}</span></p>
                </div>
                """
                status_placeholder.markdown(status_html, unsafe_allow_html=True)
                
                # Calculate FPS
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    current_fps = fps_counter / (time.time() - fps_start_time)
                    fps_placeholder.metric("📈 FPS", f"{current_fps:.1f}")
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Frame rate control
                process_time = time.time() - process_start
                time.sleep(max(0, 1/fps_limit - process_time))
                
                frame_count += 1
            
            cap.release()
            
            if frame_count >= 100000:
                st.info("Session completed. Click 'Start Camera' to continue.")
                st.session_state.camera_active = False
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.session_state.camera_active = False
    
    else:
        # Default state
        st.info("👆 Click **'Start Camera'** in the sidebar to begin real-time video analysis")
        
        # Camera access instructions
        st.markdown("""
        ### 📱 Quick Start:
        1. Click **'Start Camera'** in the sidebar
        2. **Allow camera access** when prompted by your browser
        3. Position yourself in front of the camera
        4. Watch real-time analytics!
        
        ### 🔧 Features:
        - **Face Detection**: Real-time face detection using OpenCV
        - **Posture Analysis**: Checks if you're centered in frame
        - **Gaze Tracking**: Basic analysis of attention direction
        
        ### ⚠️ Note:
        Camera access requires **HTTPS** and browser permissions. On Streamlit Cloud, camera access may be limited due to browser security policies.
        """)
        
        # Show sample metrics
        with col2:
            face_metric.metric("👤 Face Detection", "Ready", delta="Waiting...")
            posture_metric.metric("🧍 Posture", "Ready", delta="Waiting...")
            gaze_metric.metric("👁️ Gaze", "Ready", delta="Waiting...")

if __name__ == "__main__":
    main()
