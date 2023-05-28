from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

import cv2

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# Open the video file
og_video_path = "//Users//rkdbg//Codes//CV//omg_trainer//omg.mp4"
og_video_capture = cv2.VideoCapture(og_video_path)

my_video_path = "//Users//rkdbg//Codes//CV//omg_trainer//omg.mp4"
my_video_capture = cv2.VideoCapture(my_video_path)

landmark_losses = np.empty((0, 33), float)

# Check if the video file is successfully opened
if not og_video_capture.isOpened() and not my_video_capture.isOpened():
    print("Error opening video file")
    exit()

# Read the video frames until the end of the video
while og_video_capture.isOpened() and my_video_capture.isOpened():
    # Read a single frame from the video
    og_ret, og_frame = og_video_capture.read()
    my_ret, my_frame = my_video_capture.read()

    # If the frame is not read properly, break the loop
    if not og_ret or not my_ret:
        break

    # Convert the frame to RGB
    og_frame_rgb = mp.Image(mp.ImageFormat.SRGB, og_frame)
    my_frame_rgb = mp.Image(mp.ImageFormat.SRGB, my_frame)

    # Process the frame and get the pose landmarks
    og_detection_result = detector.detect(og_frame_rgb)
    my_detection_result = detector.detect(my_frame_rgb)

    og_annotated_image = draw_landmarks_on_image(og_frame_rgb.numpy_view(), og_detection_result)
    my_annotated_image = draw_landmarks_on_image(my_frame_rgb.numpy_view(), my_detection_result)

    # Draw the pose landmarks on the frame (example visualization)
    if len(og_detection_result.pose_landmarks) > 0 and len(my_detection_result.pose_landmarks) > 0:
        og_pose_landmarks = og_detection_result.pose_landmarks[0]
        my_pose_landmarks = my_detection_result.pose_landmarks[0]

        landmark_loss = []
        for og_landmark, my_landmark in zip(og_pose_landmarks, my_pose_landmarks):
            og = np.array([og_landmark.x, og_landmark.y, og_landmark.z, og_landmark.visibility])
            my = np.array([my_landmark.x, my_landmark.y, my_landmark.z, my_landmark.visibility])
            loss = np.mean((og - my)**2)
            landmark_loss.append(loss)

        landmark_losses = np.vstack((landmark_losses, np.array([landmark_loss])))

    else:
       og_pose_landmarks = 0
       my_pose_landmarks = 0
        
    # Display the resulting frame
    cv2.imshow('OG', og_annotated_image)
    cv2.imshow('MY', my_annotated_image)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(landmark_losses.shape)
        break

# Release the video capture and close the window
og_video_capture.release()
cv2.destroyAllWindows()

# # STEP 3: Load the input image.
og_image = mp.Image.create_from_file("//Users//rkdbg//Codes//CV//omg_trainer//x1080.jpg")
# my_image = mp.Image.create_from_file("//Users//rkdbg//Codes//CV//omg_trainer//스크린샷 2023-05-28 오후 11.14.15.jpeg")

# # STEP 4: Detect pose landmarks from the input image.
# og_detection_result = detector.detect(og_image)
# og_pose_landmarks = og_detection_result.pose_landmarks[0]

# my_detection_result = detector.detect(my_image)
# my_pose_landmarks = my_detection_result.pose_landmarks[0]

# landmark_losses = np.empty((0, 33), float)

# landmark_loss = []
# for og_landmark, my_landmark in zip(og_pose_landmarks, my_pose_landmarks):
#   og = np.array([og_landmark.x, og_landmark.y, og_landmark.z, og_landmark.visibility])
#   my = np.array([my_landmark.x, my_landmark.y, my_landmark.z, my_landmark.visibility])
#   loss = np.mean((og - my)**2)
#   landmark_loss.append(loss)

# landmark_losses = np.vstack((landmark_losses, np.array([landmark_loss])))
# print(landmark_losses)
# print(np.mean(landmark_losses))

# # STEP 5: Process the detection result. In this case, visualize it.
# annotated_image = draw_landmarks_on_image(og_image.numpy_view(), og_detection_result)
# my_image = draw_landmarks_on_image(my_image.numpy_view(), my_detection_result)
# cv2.imshow("og", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
# cv2.imshow("my", cv2.cvtColor(my_image, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()