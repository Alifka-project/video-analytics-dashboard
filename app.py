import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import os
from datetime import datetime
import math

# Configure Streamlit page
st.set_page_config(
    page_title="Real-time Video Analytics",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cloud deployment detection
def is_cloud_deployment():
    """Detect if running on cloud platform"""
    cloud_indicators = [
        'RAILWAY_ENVIRONMENT',  # Railway
        'RENDER',               # Render
        'DYNO',                 # Heroku
        'VERCEL',              # Vercel
    ]
    return any(os.getenv(indicator) for indicator in cloud_indicators)

# Initialize MediaPipe with cloud-optimized settings
@st.cache_resource
def load_mediapipe_models():
    mp_face_detection = mp.solutions.face_detection
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    
    # Optimized settings for cloud deployment
    confidence = 0.6 if is_cloud_deployment() else 0.5
    
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=confidence
    )
    pose = mp_pose.Pose(
        min_detection_confidence=confidence, 
        min_tracking_confidence=confidence,
        model_complexity=1  # Reduced complexity for cloud
    )
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=confidence,
        min_tracking_confidence=confidence
    )
    
    return face_detection, pose, face_mesh, mp_drawing

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

def analyze_gaze_direction(landmarks, image_shape):
    """Analyze gaze direction using facial landmarks"""
    try:
        # Get key facial landmarks for gaze estimation
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        h, w = image_shape[:2]
        
        # Calculate eye centers
        left_eye_points = [(landmarks[i].x * w, landmarks[i].y * h) for i in left_eye_indices]
        right_eye_points = [(landmarks[i].x * w, landmarks[i].y * h) for i in right_eye_indices]
        
        left_eye_center = np.mean(left_eye_points, axis=0)
        right_eye_center = np.mean(right_eye_points, axis=0)
        
        # Calculate nose tip and center
        nose_tip = (landmarks[1].x * w, landmarks[1].y * h)
        nose_center = (landmarks[168].x * w, landmarks[168].y * h)
        
        # Simple gaze estimation based on eye-nose relationship
        eye_center = ((left_eye_center[0] + right_eye_center[0]) / 2, 
                     (left_eye_center[1] + right_eye_center[1]) / 2)
        
        # Calculate horizontal deviation
        horizontal_deviation = abs(eye_center[0] - nose_center[0])
        
        # Threshold for "looking at camera"
        looking_threshold = 20  # Slightly higher for cloud deployment
        
        return horizontal_deviation < looking_threshold, horizontal_deviation
    
    except Exception as e:
        return False, 0

def analyze_posture(pose_landmarks, image_shape):
    """Analyze if posture is centered"""
    try:
        h, w = image_shape[:2]
        
        # Get shoulder landmarks
        left_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Calculate shoulder center
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2 * w
        
        # Calculate deviation from image center
        image_center_x = w / 2
        deviation = abs(shoulder_center_x - image_center_x)
        
        # Threshold for centered posture
        centered_threshold = w * 0.12  # Slightly more lenient for cloud
        
        is_centered = deviation < centered_threshold
        deviation_percentage = (deviation / (w / 2)) * 100
        
        return is_centered, deviation_percentage
    
    except Exception as e:
        return False, 100

def process_frame(frame, face_detection, pose, face_mesh, mp_drawing):
    """Process a single frame and return analytics"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Initialize results
    face_visible = False
    posture_centered = False
    looking_at_camera = False
    posture_deviation = 100
    gaze_deviation = 100
    
    # Face detection
    face_results = face_detection.process(rgb_frame)
    if face_results.detections:
        face_visible = True
        
        # Draw face detection
        for detection in face_results.detections:
            mp_drawing.draw_detection(frame, detection)
    
    # Pose detection
    pose_results = pose.process(rgb_frame)
    if pose_results.pose_landmarks:
        # Draw pose landmarks (simplified for performance)
        mp_drawing.draw_landmarks(
            frame, pose_results.pose_landmarks, 
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)
        )
        
        # Analyze posture
        posture_centered, posture_deviation = analyze_posture(
            pose_results.pose_landmarks, frame.shape
        )
    
    # Face mesh for gaze detection (simplified for cloud performance)
    face_mesh_results = face_mesh.process(rgb_frame)
    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            # Analyze gaze
            looking_at_camera, gaze_deviation = analyze_gaze_direction(
                face_landmarks.landmark, frame.shape
            )
    
    return frame, face_visible, posture_centered, looking_at_camera, posture_deviation, gaze_deviation

def main():
    # Header with deployment info
    if is_cloud_deployment():
        st.success("🌐 Running on Cloud Platform - Camera access available via HTTPS!")
    else:
        st.info("💻 Running locally")
    
    st.title("🎥 Real-time Video Analytics Dashboard")
    st.markdown("---")
    
    # Load MediaPipe models
    face_detection, pose, face_mesh, mp_drawing = load_mediapipe_models()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Settings")
    
    # Cloud-optimized default settings
    default_width = 480 if is_cloud_deployment() else 640
    default_height = 360 if is_cloud_deployment() else 480
    default_fps = 12 if is_cloud_deployment() else 15
    
    # Camera settings
    camera_index = st.sidebar.selectbox("Camera Index", [0, 1, 2], index=0)
    frame_width = st.sidebar.slider("Frame Width", 320, 1280, default_width, 160)
    frame_height = st.sidebar.slider("Frame Height", 240, 720, default_height, 120)
    
    # Analytics thresholds
    st.sidebar.subheader("Analytics Thresholds")
    face_confidence = st.sidebar.slider("Face Detection Confidence", 0.1, 1.0, 0.6, 0.1)
    posture_threshold = st.sidebar.slider("Posture Center Threshold (%)", 5, 25, 12, 1)
    gaze_threshold = st.sidebar.slider("Gaze Center Threshold (px)", 5, 30, 20, 1)
    
    # Performance settings
    st.sidebar.subheader("Performance")
    fps_limit = st.sidebar.slider("FPS Limit", 5, 30, default_fps, 1)
    
    # Cloud deployment tips
    if is_cloud_deployment():
        st.sidebar.info("💡 Cloud Tips:\n- Use lower resolution for better performance\n- Reduce FPS if experiencing lag\n- Allow camera access in browser")
    
    # Start/Stop button
    start_analytics = st.sidebar.button("🚀 Start Analytics", type="primary")
    stop_analytics = st.sidebar.button("⏹️ Stop Analytics")
    
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
        st.subheader("🚦 Status Indicators")
        status_placeholder = st.empty()
        
        # Performance metrics
        st.subheader("⚡ Performance")
        fps_placeholder = st.empty()
        latency_placeholder = st.empty()
    
    # Initialize session state
    if 'analytics_running' not in st.session_state:
        st.session_state.analytics_running = False
    
    if start_analytics:
        st.session_state.analytics_running = True
    
    if stop_analytics:
        st.session_state.analytics_running = False
    
    # Main analytics loop
    if st.session_state.analytics_running:
        try:
            # Initialize camera
            cap = cv2.VideoCapture(camera_index)
            
            if not cap.isOpened():
                st.error("❌ Could not open camera. Please check:")
                st.error("1. Camera permissions are granted")
                st.error("2. No other apps are using the camera")
                st.error("3. Try different camera index (0, 1, 2)")
                return
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
            cap.set(cv2.CAP_PROP_FPS, fps_limit)
            
            # Performance tracking
            frame_count = 0
            start_time = time.time()
            fps_counter = 0
            fps_start_time = time.time()
            
            # Processing loop
            while st.session_state.analytics_running:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to read from camera")
                    break
                
                # Process frame
                process_start = time.time()
                
                processed_frame, face_visible, posture_centered, looking_at_camera, posture_deviation, gaze_deviation = process_frame(
                    frame, face_detection, pose, face_mesh, mp_drawing
                )
                
                process_time = time.time() - process_start
                
                # Display video
                video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
                
                # Update metrics
                face_metric.metric(
                    "👤 Face Detection",
                    "Visible" if face_visible else "Not Visible",
                    delta="✅ Detected" if face_visible else "❌ Not found"
                )
                
                posture_metric.metric(
                    "🧍 Posture Analysis",
                    "Centered" if posture_centered else "Off-center",
                    delta=f"{posture_deviation:.1f}% deviation"
                )
                
                gaze_metric.metric(
                    "👁️ Gaze Tracking",
                    "Looking at camera" if looking_at_camera else "Looking away",
                    delta=f"{gaze_deviation:.1f}px deviation"
                )
                
                # Status indicators
                status_html = f"""
                <div style="padding: 15px; border-radius: 10px; background-color: #f0f2f6;">
                    <h4>Current Status:</h4>
                    <p><strong>👤 Face:</strong> <span style="color: {'green' if face_visible else 'red'}; font-size: 18px;">{'✅ Visible' if face_visible else '❌ Not Detected'}</span></p>
                    <p><strong>🧍 Posture:</strong> <span style="color: {'green' if posture_centered else 'orange'}; font-size: 18px;">{'✅ Centered' if posture_centered else '⚠️ Off-Center'}</span></p>
                    <p><strong>👁️ Gaze:</strong> <span style="color: {'green' if looking_at_camera else 'red'}; font-size: 18px;">{'✅ On Camera' if looking_at_camera else '❌ Looking Away'}</span></p>
                </div>
                """
                status_placeholder.markdown(status_html, unsafe_allow_html=True)
                
                # Calculate and display FPS
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    current_fps = fps_counter / (time.time() - fps_start_time)
                    fps_placeholder.metric("📈 Current FPS", f"{current_fps:.1f}")
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Display processing latency
                latency_ms = process_time * 1000
                latency_color = "green" if latency_ms < 100 else "orange" if latency_ms < 200 else "red"
                latency_placeholder.metric("⏱️ Processing Time", f"{latency_ms:.1f} ms")
                
                # Frame rate control
                time.sleep(max(0, 1/fps_limit - process_time))
                
                frame_count += 1
            
            cap.release()
            st.success("✅ Analytics stopped successfully")
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.error("Please try refreshing the page or adjusting settings")
            st.session_state.analytics_running = False
    
    else:
        # Default state
        st.info("👆 Click **'Start Analytics'** in the sidebar to begin real-time video analysis")
        
        # Instructions for cloud deployment
        if is_cloud_deployment():
            st.markdown("""
            ### 📱 Quick Start Instructions:
            1. Click **'Start Analytics'** in the sidebar
            2. Allow camera access when prompted by your browser
            3. Position yourself in front of the camera
            4. Watch real-time analytics update!
            
            ### 🔧 Having issues?
            - Ensure you're using **HTTPS** (this deployment should automatically provide it)
            - Try different browsers (Chrome/Edge work best)
            - Check camera permissions in browser settings
            - Reduce frame size/FPS if performance is slow
            """)
        
        # Show sample metrics when not running
        with col2:
            face_metric.metric("👤 Face Detection", "Ready to start", delta="Waiting...")
            posture_metric.metric("🧍 Posture Analysis", "Ready to start", delta="Waiting...")
            gaze_metric.metric("👁️ Gaze Tracking", "Ready to start", delta="Waiting...")
            
            # Sample status
            status_html = """
            <div style="padding: 15px; border-radius: 10px; background-color: #e8f4ff;">
                <h4>System Ready!</h4>
                <p>🟡 Waiting to start analytics...</p>
                <p>📹 Camera: Standby</p>
                <p>🤖 AI Models: Loaded</p>
            </div>
            """
            status_placeholder.markdown(status_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
