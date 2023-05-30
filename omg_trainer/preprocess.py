import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def draw_landmarks_on_image(rgb_image, detection_result):
  if detection_result is None:
    return rgb_image
  
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

prac_landmarker_result = None
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global prac_landmarker_result
    prac_landmarker_result = result

og_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='omg_trainer//pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.VIDEO)

prac_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='omg_trainer//pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

og_landmarker = PoseLandmarker.create_from_options(og_options)
prac_landmarker = PoseLandmarker.create_from_options(prac_options)

def frame_with_landmark(frame, timestamp):
   og_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
   og_landmarker_result = og_landmarker.detect_for_video(og_image, timestamp)

   return draw_landmarks_on_image(og_image.numpy_view(), og_landmarker_result)

def webcam_frame_with_landmark(frame, timestamp):
    prac_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    prac_landmarker.detect_async(prac_image, timestamp)

    return draw_landmarks_on_image(prac_image.numpy_view(), prac_landmarker_result)