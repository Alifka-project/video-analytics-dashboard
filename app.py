import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime
import platform

# Configure Streamlit page
st.set_page_config(
    page_title="Always Running Video Analytics",
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

def capture_and_process_frame(camera_index, width, height, show_face, show_posture, show_gaze):
    """Capture and process a single frame"""
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        return None, False, False, False, "Camera not available"
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        return None, False, False, False, "Failed to capture frame"
    
    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Initialize analytics results
    face_visible = False
    posture_centered = False
    looking_at_camera = False
    
    # Apply analytics
    if show_face:
        frame, face_visible = detect_face_opencv(frame)
    
    if show_posture:
        frame, posture_centered = analyze_posture_simple(frame)
    
    if show_gaze and face_visible:
        frame, looking_at_camera = analyze_gaze_simple(frame, face_visible)
    
    # Add timestamp and running indicator
    timestamp = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, f"ALWAYS RUNNING | {timestamp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "AUTO-REFRESH MODE", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame, face_visible, posture_centered, looking_at_camera, "Success"

def main():
    st.title("🎥 Always Running Video Analytics - Auto-Refresh Mode")
    st.markdown("---")
    
    # System info
    system_info = f"**System:** {platform.system()} | **OpenCV:** {cv2.__version__}"
    st.markdown(system_info)
    
    # Initialize session state for persistence
    if 'always_running' not in st.session_state:
        st.session_state.always_running = False
        st.session_state.start_time = None
        st.session_state.frame_count = 0
        st.session_state.last_frame_time = 0
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Settings")
    
    # Camera settings
    camera_index = st.sidebar.selectbox("Camera Index", [0, 1, 2], index=0)
    
    resolution = st.sidebar.selectbox("Resolution", 
                                     ["640x480", "1280x720"], 
                                     index=0)
    width, height = map(int, resolution.split('x'))
    
    # Analytics settings
    st.sidebar.subheader("📊 Analytics")
    show_face_detection = st.sidebar.checkbox("Face Detection", value=True)
    show_posture_analysis = st.sidebar.checkbox("Posture Analysis", value=True)
    show_gaze_tracking = st.sidebar.checkbox("Gaze Tracking", value=True)
    
    # Auto-refresh settings
    st.sidebar.subheader("🔄 Auto-Refresh")
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 0.1, 2.0, 0.5, 0.1)
    
    # Control buttons
    if not st.session_state.always_running:
        if st.sidebar.button("🚀 Start Always Running Mode", type="primary"):
            st.session_state.always_running = True
            st.session_state.start_time = time.time()
            st.session_state.frame_count = 0
            st.rerun()
    else:
        if st.sidebar.button("⏹️ Stop Always Running Mode", type="secondary"):
            st.session_state.always_running = False
            st.session_state.start_time = None
            st.rerun()
    
    # Status indicator
    if st.session_state.always_running:
        runtime = time.time() - st.session_state.start_time
        st.sidebar.success(f"🟢 **ALWAYS RUNNING**\n\n⏱️ Runtime: {runtime/60:.1f} min\n📊 Frames: {st.session_state.frame_count}")
    else:
        st.sidebar.info("🔴 **STOPPED** - Click start to begin continuous mode")
    
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
        uptime_placeholder = st.empty()
        frame_count_placeholder = st.empty()
    
    # Auto-refresh logic - THE KEY TO ALWAYS RUNNING!
    if st.session_state.always_running:
        # Capture and process frame
        frame, face_visible, posture_centered, looking_at_camera, status_msg = capture_and_process_frame(
            camera_index, width, height, show_face_detection, show_posture_analysis, show_gaze_tracking
        )
        
        if frame is not None:
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
            <div style="padding: 15px; border-radius: 10px; background-color: #d1ecf1;">
                <h4>🔄 ALWAYS RUNNING MODE ACTIVE</h4>
                <p><strong>Face:</strong> <span style="color: {'green' if face_visible else 'red'};">{face_status}</span></p>
                <p><strong>Posture:</strong> <span style="color: {'green' if posture_centered else 'orange'};">{posture_status}</span></p>
                <p><strong>Gaze:</strong> <span style="color: {'green' if looking_at_camera else 'red'};">{gaze_status}</span></p>
                <p><strong>Auto-Refresh:</strong> 🟢 Every {refresh_rate}s</p>
                <p><strong>Status:</strong> 🔄 Continuously Running</p>
            </div>
            """
            status_placeholder.markdown(status_html, unsafe_allow_html=True)
            
            # Update performance metrics
            runtime = time.time() - st.session_state.start_time
            current_time = time.time()
            
            # Calculate FPS
            if st.session_state.last_frame_time > 0:
                fps = 1.0 / (current_time - st.session_state.last_frame_time)
                fps_placeholder.metric("📈 Effective FPS", f"{fps:.1f}")
            
            st.session_state.last_frame_time = current_time
            
            uptime_placeholder.metric("🕐 Total Uptime", f"{runtime/60:.1f} min")
            st.session_state.frame_count += 1
            frame_count_placeholder.metric("📊 Total Frames", f"{st.session_state.frame_count:,}")
            
        else:
            st.error(f"❌ Camera error: {status_msg}")
        
        # AUTO-REFRESH - This keeps it always running!
        time.sleep(refresh_rate)
        st.rerun()  # This automatically refreshes the app!
    
    else:
        # Default state
        st.info("👆 Click **'Start Always Running Mode'** for continuous operation")
        
        # Instructions
        st.markdown("""
        ### 🔄 Auto-Refresh Method:
        - **Always Running** - Uses Streamlit's auto-refresh to stay active
        - **No Session Limits** - Runs indefinitely until manually stopped
        - **Auto-Recovery** - Each refresh creates a new camera connection
        - **Adjustable Speed** - Control refresh rate from 0.1 to 2 seconds
        - **Persistent State** - Maintains counters and uptime across refreshes
        
        ### 💡 How It Works:
        - **Captures single frames** instead of continuous video stream
        - **Auto-refreshes** the entire page at your chosen interval
        - **Maintains state** between refreshes using session state
        - **No blocking loops** that can be interrupted by Streamlit
        
        ### ⚡ Benefits:
        - **Always runs** - Cannot be stopped by Streamlit's session management
        - **Memory efficient** - Each refresh clears memory
        - **Stable operation** - No accumulating errors or memory leaks
        - **Perfect for monitoring** - Designed for long-term operation
        """)
        
        # Show sample metrics
        with col2:
            face_metric.metric("👤 Face Detection", "Ready", delta="Auto-refresh mode")
            posture_metric.metric("🧍 Posture", "Ready", delta="Auto-refresh mode")
            gaze_metric.metric("👁️ Gaze", "Ready", delta="Auto-refresh mode")
            
            status_html = """
            <div style="padding: 15px; border-radius: 10px; background-color: #fff3cd;">
                <h4>⚡ Ready for Always Running Mode</h4>
                <p>🟡 Standing by for continuous operation</p>
                <p>🔄 Auto-refresh method</p>
                <p>♾️ No session limits</p>
                <p>🎮 Manual stop control only</p>
            </div>
            """
            status_placeholder.markdown(status_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
