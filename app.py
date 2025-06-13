import streamlit as st
import cv2
import numpy as np
import time
import mediapipe as mp
import platform
import sys

# Initialize MediaPipe solutions
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


# Face detection function
def detect_faces(image):
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        faces = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                faces.append(bbox)

        return faces


# Eye tracking function
def detect_eyes(image):
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        eye_landmarks = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Eye landmarks indices
                left_eye = [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
                right_eye = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]

                h, w, _ = image.shape
                left_eye_px = [(int(l.x * w), int(l.y * h)) for l in left_eye]
                right_eye_px = [(int(l.x * w), int(l.y * h)) for l in right_eye]

                eye_landmarks.append((left_eye_px, right_eye_px))

        return eye_landmarks


# Check if face is centered
def is_face_centered(image, face_bbox, threshold=0.2):
    h, w, _ = image.shape
    x, y, width, height = face_bbox

    face_center_x = x + width // 2
    face_center_y = y + height // 2

    img_center_x = w // 2
    img_center_y = h // 2

    x_deviation = abs(face_center_x - img_center_x) / w
    y_deviation = abs(face_center_y - img_center_y) / h

    return x_deviation <= threshold and y_deviation <= threshold, (x_deviation, y_deviation)


# Draw feedback on image
def draw_feedback(image, faces, eye_landmarks=None):
    h, w, _ = image.shape
    img_center_x = w // 2
    img_center_y = h // 2

    # Draw center crosshair
    cv2.line(image, (img_center_x, 0), (img_center_x, h), (0, 255, 0), 1)
    cv2.line(image, (0, img_center_y), (w, img_center_y), (0, 255, 0), 1)

    if len(faces) == 0:
        cv2.putText(image, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return image

    if len(faces) > 1:
        cv2.putText(image, "Multiple faces detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        face_center_x = x + w // 2
        face_center_y = y + h // 2

        # Draw face center
        cv2.circle(image, (face_center_x, face_center_y), 5, (255, 0, 0), -1)

        # Check if face is centered
        centered, (x_dev, y_dev) = is_face_centered(image, (x, y, w, h))

        if centered:
            cv2.putText(image, "Face centered", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(image, "Center your face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Add directional guidance
            if face_center_x < img_center_x - 0.1 * w:
                cv2.putText(image, "Move right", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif face_center_x > img_center_x + 0.1 * w:
                cv2.putText(image, "Move left", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if face_center_y < img_center_y - 0.1 * h:
                cv2.putText(image, "Move down", (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif face_center_y > img_center_y + 0.1 * h:
                cv2.putText(image, "Move up", (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Draw eye landmarks if available
    if eye_landmarks:
        for (left_eye, right_eye) in eye_landmarks:
            # Draw left eye
            for point in left_eye:
                cv2.circle(image, point, 2, (0, 255, 255), -1)

            # Draw right eye
            for point in right_eye:
                cv2.circle(image, point, 2, (0, 255, 255), -1)

    return image


# Check system requirements
def check_system_requirements():
    requirements = {
        "OS": platform.system() + " " + platform.version(),
        "Python": sys.version.split()[0],
        "OpenCV": cv2.__version__,
        "MediaPipe": mp.__version__,
        "Streamlit": st.__version__,
        "Camera": "Checking...",
    }

    # Check if camera is available
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        requirements["Camera"] = "Available"
        cap.release()
    else:
        requirements["Camera"] = "Not Available"

    return requirements


# Get version information
def get_version_info():
    version_info = {
        "app_version": "1.0.0",
        "last_updated": "2023-11-07",
        "features": [
            "Face detection",
            "Face centering guidance",
            "Eye tracking",
            "Multiple face warning",
            "System compatibility check"
        ]
    }
    return version_info


def main():
    # Set page configuration
    st.set_page_config(
        page_title="Interview Ready - Camera Check",
        page_icon="📹",
        layout="wide"
    )

    # Add CSS styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #424242;
            margin-bottom: 1rem;
        }
        .info-text {
            font-size: 1rem;
            color: #616161;
        }
        .success-text {
            color: #4CAF50;
            font-weight: bold;
        }
        .error-text {
            color: #F44336;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #1E88E5;
            color: white;
            font-weight: bold;
            border-radius: 5px;
            border: none;
            padding: 0.5rem 1rem;
            margin-top: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">Interview Ready - Camera Check</h1>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Ensure your camera setup is perfect for your upcoming interview!</p>',
                unsafe_allow_html=True)

    # Sidebar with version info
    st.sidebar.markdown('<h2 class="sub-header">About</h2>', unsafe_allow_html=True)
    version_info = get_version_info()
    st.sidebar.markdown(f"**App Version:** {version_info['app_version']}")
    st.sidebar.markdown(f"**Last Updated:** {version_info['last_updated']}")

    st.sidebar.markdown('<h2 class="sub-header">Features</h2>', unsafe_allow_html=True)
    for feature in version_info['features']:
        st.sidebar.markdown(f"- {feature}")

    # System Requirements
    st.sidebar.markdown('<h2 class="sub-header">System Check</h2>', unsafe_allow_html=True)
    requirements = check_system_requirements()

    for key, value in requirements.items():
        if key == "Camera" and value == "Not Available":
            st.sidebar.markdown(f"**{key}:** <span class='error-text'>{value}</span>", unsafe_allow_html=True)
        else:
            st.sidebar.markdown(f"**{key}:** {value}")

    # Main content with tabs
    tabs = st.tabs(["Camera Check", "Guidelines", "Help"])

    with tabs[0]:
        st.markdown('<h2 class="sub-header">Camera Check</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            # Camera feed
            FRAME_WINDOW = st.empty()

            # Camera control buttons
            start_button = st.button("Start Camera")
            stop_button = st.button("Stop Camera")

            if 'camera_running' not in st.session_state:
                st.session_state.camera_running = False

            if start_button:
                st.session_state.camera_running = True

            if stop_button:
                st.session_state.camera_running = False

            if st.session_state.camera_running:
                cap = cv2.VideoCapture(0)

                if not cap.isOpened():
                    st.error("Error: Could not open camera. Please check your camera connection.")
                else:
                    while st.session_state.camera_running:
                        ret, frame = cap.read()

                        if not ret:
                            st.error("Error: Failed to capture image from camera.")
                            break

                        # Process the frame
                        faces = detect_faces(frame)
                        eye_landmarks = detect_eyes(frame)
                        processed_frame = draw_feedback(frame.copy(), faces, eye_landmarks)

                        # Display the frame
                        FRAME_WINDOW.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB")

                        # Limit frame rate
                        time.sleep(0.03)

                    cap.release()
            else:
                # Display placeholder image
                placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder_img, "Click 'Start Camera' to begin", (100, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                FRAME_WINDOW.image(placeholder_img, channels="RGB")

        with col2:
            st.markdown('<h3 class="sub-header">Guidelines</h3>', unsafe_allow_html=True)
            st.markdown("""
            - Keep your face centered in the frame
            - Ensure good lighting on your face
            - Look directly at the camera
            - Only one person should be visible
            - Avoid busy backgrounds
            """)

            st.markdown('<h3 class="sub-header">Status</h3>', unsafe_allow_html=True)

            if st.session_state.camera_running:
                st.markdown('<p class="success-text">✅ Camera is running</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="error-text">❌ Camera is off</p>', unsafe_allow_html=True)

    with tabs[1]:
        st.markdown('<h2 class="sub-header">Interview Camera Guidelines</h2>', unsafe_allow_html=True)

        st.markdown("""
        ### Face Positioning
        - Center your face in the frame
        - Your face should occupy approximately 30% of the screen
        - Keep your eyes at the upper third of the frame

        ### Lighting
        - Ensure your face is well-lit
        - Avoid backlighting (don't sit with a window behind you)
        - Natural, diffused light works best

        ### Camera Angle
        - Camera should be at eye level
        - Look directly into the camera when speaking
        - Avoid looking down at the camera

        ### Background
        - Choose a clean, professional background
        - Avoid busy or distracting elements
        - Consider using a neutral wall or simple virtual background
        """)

    with tabs[2]:
        st.markdown('<h2 class="sub-header">Help & Troubleshooting</h2>', unsafe_allow_html=True)

        st.markdown("""
        ### Common Issues

        #### Camera not working
        - Check if your camera is properly connected
        - Make sure no other application is using your camera
        - Restart your browser or computer
        - Check camera permissions in your browser

        #### Face not detected
        - Improve lighting conditions - your face should be clearly visible
        - Position yourself directly in front of the camera
        - Remove any accessories that might obscure your face

        #### Multiple faces detected
        - Ensure you are the only person visible in the frame
        - Check for photos or pictures in the background that might be detected as faces
        """)

        # FAQ Expanders
        st.markdown('<h3 class="sub-header">Frequently Asked Questions</h3>', unsafe_allow_html=True)

        with st.expander("How do I know if my setup is ready for the interview?"):
            st.write("""
            Your setup is ready when:
            1. Your face is clearly visible and centered in the frame
            2. Only one face (yours) is detected
            3. The lighting is good, and your features are clearly visible
            4. You can maintain eye contact by looking at the camera
            5. Your background is professional and not distracting
            """)


if __name__ == "__main__":
    main()
