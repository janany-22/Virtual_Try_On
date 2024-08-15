import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe for human pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, 
                    model_complexity=1, 
                    smooth_landmarks=True, 
                    enable_segmentation=False, 
                    min_detection_confidence=0.5, 
                    min_tracking_confidence=0.5)

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load the dress image
dress_image = cv2.imread('content\\Shirts\\2.png', cv2.IMREAD_UNCHANGED)

if dress_image is None:
    print("Error: Could not load the dress image.")
    exit()

def background_removal(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def visualize_pose(image, landmarks):
    mp_drawing.draw_landmarks(
        image, 
        landmarks, 
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )

def user_segmentation(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        visualize_pose(image, results.pose_landmarks)
        return image, results.pose_landmarks
    else:
        return image, None

def overlay_dress_on_user(user_image, dress_image, landmarks):
    if landmarks is None:
        return user_image

    # Get shoulder points
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    # Calculate dress position and size
    shoulder_distance = int(abs(left_shoulder.x - right_shoulder.x) * user_image.shape[1])
    dress_height = max(1, int(shoulder_distance * 2.8))  # Ensure minimum height of 1
    dress_width = max(1, int(shoulder_distance * 2.5))   # Ensure minimum width of 1

    # Resize dress image
    try:
        resized_dress = cv2.resize(dress_image, (dress_width, dress_height))
    except cv2.error:
        print(f"Failed to resize dress. Width: {dress_width}, Height: {dress_height}")
        return user_image  # Return original image if resizing fails

    # Calculate dress position
    x_offset = int(min(left_shoulder.x, right_shoulder.x) * user_image.shape[1])
    y_offset = int(min(left_shoulder.y, right_shoulder.y) * user_image.shape[0])

    # Ensure the dress fits within the frame
    if y_offset + dress_height > user_image.shape[0]:
        dress_height = max(1, user_image.shape[0] - y_offset)
    if x_offset + dress_width > user_image.shape[1]:
        dress_width = max(1, user_image.shape[1] - x_offset)

    # Resize again if dimensions changed
    try:
        resized_dress = cv2.resize(resized_dress, (dress_width, dress_height))
    except cv2.error:
        print(f"Failed to resize dress. Width: {dress_width}, Height: {dress_height}")
        return user_image  # Return original image if resizing fails

    # Create a mask for the dress
    if resized_dress.shape[2] == 4:  # Check if the dress image has an alpha channel
        mask = resized_dress[:, :, 3] / 255.0
    else:
        mask = np.ones((dress_height, dress_width), dtype=np.float32)
    mask_inv = 1.0 - mask

    # Cut out the area where the dress will be placed
    roi = user_image[y_offset:y_offset+dress_height, x_offset:x_offset+dress_width]

    # Ensure ROI and resized_dress have the same dimensions
    if roi.shape[:2] != resized_dress.shape[:2]:
        print("ROI and resized dress dimensions do not match.")
        return user_image

    # Now blend the dress with the ROI
    for c in range(0, 3):
        roi[:, :, c] = roi[:, :, c] * mask_inv + resized_dress[:, :, c] * mask

    # Put the blended region back into the image
    user_image[y_offset:y_offset+dress_height, x_offset:x_offset+dress_width] = roi

    return user_image

def virtual_try_on(video_source=0):
    cap = cv2.VideoCapture(video_source)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        
        # Step 1: Remove background
        frame_no_bg = background_removal(frame)
        
        # Step 2: Segment the user and get pose landmarks
        segmented_user, landmarks = user_segmentation(frame_no_bg)
        
        # Step 3: Overlay dress on the user
        result = overlay_dress_on_user(segmented_user, dress_image, landmarks)
        
        # Display the final result
        cv2.imshow('Virtual Try-On', result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

virtual_try_on()
