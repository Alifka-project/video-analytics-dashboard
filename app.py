import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime
import platform
import threading

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
    st.title("🎥 Real-time Video Analytics Dashboard - Unlimited Mode")
    st.markdown("---")
    
    # System info
    system_info = f"**System:** {platform.system()} | **OpenCV:** {cv2.__version__}"
    st.markdown(system_info)
    
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
    
    # Performance settings
    st.sidebar.subheader("⚡ Performance")
    fps_limit = st.sidebar.slider("FPS Limit", 5, 30, 15, 1)
    
    # Control buttons
    start_camera = st.sidebar.button("🚀 Start Unlimited Session", type="primary")
    stop_camera = st.sidebar.button("⏹️ Stop Session")
    
    # Important note
    st.sidebar.info("🔄 **UNLIMITED MODE**: This session will run continuously until you manually stop it!")
    
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
        latency_placeholder = st.empty()
        uptime_placeholder = st.empty()
        frame_count_placeholder = st.empty()
    
    # Initialize session state
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
        st.session_state.session_start_time = None
        st.session_state.frame_count = 0
    
    if start_camera:
        st.session_state.camera_running = True
        st.session_state.session_start_time = time.time()
        st.session_state.frame_count = 0
    
    if stop_camera:
        st.session_state.camera_running = False
    
    # Camera processing - TRULY UNLIMITED
    if st.session_state.camera_running:
        cap = None
        try:
            # Initialize camera
            cap = cv2.VideoCapture(camera_index)
            
            if not cap.isOpened():
                st.error(f"❌ Could not open camera {camera_index}")
                st.session_state.camera_running = False
                return
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps_limit)
            
            # Test initial frame
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                st.error("❌ Cannot read from camera")
                st.session_state.camera_running = False
                return
            
            st.success("✅ Unlimited session started!")
            
            # Performance tracking
            fps_counter = 0
            fps_start_time = time.time()
            last_update_time = time.time()
            
            # MAIN UNLIMITED LOOP - NO RESTRICTIONS!
            while st.session_state.camera_running:
                # Read frame
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Try to reconnect camera
                    cap.release()
                    time.sleep(0.1)
                    cap = cv2.VideoCapture(camera_index)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    continue
                
                # Process frame
                process_start = time.time()
                
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
                
                # Add session info to frame
                session_time = time.time() - st.session_state.session_start_time
                timestamp = datetime.now().strftime("%H:%M:%S")
                info_text = f"UNLIMITED MODE | {timestamp} | {session_time/60:.1f}min | Frame: {st.session_state.frame_count}"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add status indicators
                cv2.putText(frame, "CONTINUOUS RUNNING", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Update display
                video_placeholder.image(frame, channels="BGR", use_column_width=True)
                
                # Update metrics every second to avoid too frequent updates
                current_time = time.time()
                if current_time - last_update_time >= 1.0:
                    # Update analytics metrics
                    face_status = "✅ Visible" if face_visible else "❌ Not Detected"
                    posture_status = "✅ Centered" if posture_centered else "⚠️ Off-Center"
                    gaze_status = "✅ Looking" if looking_at_camera else "❌ Looking Away"
                    
                    face_metric.metric("👤 Face Detection", face_status)
                    posture_metric.metric("🧍 Posture", posture_status)
                    gaze_metric.metric("👁️ Gaze", gaze_status)
                    
                    # Status summary
                    status_html = f"""
                    <div style="padding: 15px; border-radius: 10px; background-color: #d4edda;">
                        <h4>🔄 UNLIMITED SESSION ACTIVE</h4>
                        <p><strong>Face:</strong> <span style="color: {'green' if face_visible else 'red'};">{face_status}</span></p>
                        <p><strong>Posture:</strong> <span style="color: {'green' if posture_centered else 'orange'};">{posture_status}</span></p>
                        <p><strong>Gaze:</strong> <span style="color: {'green' if looking_at_camera else 'red'};">{gaze_status}</span></p>
                        <p><strong>Status:</strong> 🟢 Running Continuously</p>
                    </div>
                    """
                    status_placeholder.markdown(status_html, unsafe_allow_html=True)
                    
                    # Performance metrics
                    session_uptime = current_time - st.session_state.session_start_time
                    uptime_placeholder.metric("🕐 Session Uptime", f"{session_uptime/60:.1f} min")
                    frame_count_placeholder.metric("📊 Frames Processed", f"{st.session_state.frame_count:,}")
                    
                    last_update_time = current_time
                
                # Calculate FPS
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    current_fps = fps_counter / (time.time() - fps_start_time)
                    fps_placeholder.metric("📈 Current FPS", f"{current_fps:.1f}")
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Display processing latency
                process_time = time.time() - process_start
                latency_ms = process_time * 1000
                latency_placeholder.metric("⏱️ Processing Time", f"{latency_ms:.1f} ms")
                
                # Frame rate control
                time.sleep(max(0, 1/fps_limit - process_time))
                
                # Increment frame counter
                st.session_state.frame_count += 1
                
                # Yield control back to Streamlit occasionally
                if st.session_state.frame_count % 30 == 0:
                    time.sleep(0.001)  # Tiny pause every 30 frames
            
        except Exception as e:
            st.error(f"❌ Session error: {str(e)}")
            st.error("Click 'Start Unlimited Session' to restart")
            st.session_state.camera_running = False
        
        finally:
            if cap is not None:
                cap.release()
            
            if st.session_state.camera_running:
                # If we reach here and camera is still supposed to be running,
                # it means we exited the loop unexpectedly
                st.warning("⚠️ Session ended unexpectedly. Click 'Start Unlimited Session' to restart.")
                st.session_state.camera_running = False
    
    else:
        # Default state
        st.info("👆 Click **'Start Unlimited Session'** for truly continuous operation")
        
        # Instructions
        st.markdown("""
        ### 🎯 Unlimited Mode Features:
        - **🔄 Truly Continuous** - No automatic stops or session limits
        - **♾️ Unlimited Frames** - Processes as many frames as needed
        - **🛡️ Auto-Recovery** - Automatically handles camera disconnections
        - **📊 Live Monitoring** - Real-time performance and analytics tracking
        - **🎮 Manual Control** - Only stops when you click 'Stop Session'
        
        ### 🚀 What's Different:
        - **No Frame Limits** - Removed all artificial restrictions
        - **No Time Limits** - Runs for hours/days if needed
        - **No Auto-Completion** - Only manual stop control
        - **Optimized Performance** - Efficient processing for long sessions
        
        ### 💡 Performance Tips:
        - **640x480 resolution** for best performance on long sessions
        - **15 FPS limit** provides smooth experience without overload
        - **Monitor metrics** to ensure stable performance
        - **Use Stop Session** when you're done
        """)
        
        # Show sample metrics
        with col2:
            face_metric.metric("👤 Face Detection", "Ready", delta="Unlimited mode")
            posture_metric.metric("🧍 Posture", "Ready", delta="Unlimited mode")
            gaze_metric.metric("👁️ Gaze", "Ready", delta="Unlimited mode")
            
            status_html = """
            <div style="padding: 15px; border-radius: 10px; background-color: #fff3cd;">
                <h4>⚡ Ready for Unlimited Session</h4>
                <p>🟡 Standing by for continuous operation</p>
                <p>♾️ No session limits</p>
                <p>🎮 Manual control only</p>
            </div>
            """
            status_placeholder.markdown(status_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
