import mediapipe as mp
import cv2
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

def calculate_loss(landmark_losses, og_detection_result, prac_detection_result):
  if prac_detection_result is None:
     return landmark_losses
  if len(og_detection_result.pose_landmarks) > 0 and len(prac_detection_result.pose_landmarks) > 0:
    og_pose_landmarks = og_detection_result.pose_landmarks[0]
    prac_pose_landmarks = prac_detection_result.pose_landmarks[0]

    landmark_loss = []
    for og_landmark, prac_landmark in zip(og_pose_landmarks, prac_pose_landmarks):
        og = np.array([og_landmark.x, og_landmark.y, og_landmark.z, og_landmark.visibility])
        prac = np.array([prac_landmark.x, prac_landmark.y, prac_landmark.z, prac_landmark.visibility])
        loss = np.mean((og - prac)**2)
        landmark_loss.append(loss)

    landmark_losses = np.vstack((landmark_losses, np.array([landmark_loss])))

  elif len(og_detection_result.pose_landmarks) > 0:
    og_pose_landmarks = og_detection_result.pose_landmarks[0]

    landmark_loss = []
    for og_landmark in og_pose_landmarks:
        og = np.array([og_landmark.x, og_landmark.y, og_landmark.z, og_landmark.visibility])
        loss = np.mean(og**2)
        landmark_loss.append(loss)

    landmark_losses = np.vstack((landmark_losses, np.array([landmark_loss])))

  elif len(prac_detection_result.pose_landmarks) > 0:
    prac_pose_landmarks = prac_detection_result.pose_landmarks[0]

    landmark_loss = []
    for prac_landmark in prac_pose_landmarks:
      prac = np.array([prac_landmark.x, prac_landmark.y, prac_landmark.z, prac_landmark.visibility])
      loss = np.mean(prac**2)
      landmark_loss.append(loss)

    landmark_losses = np.vstack((landmark_losses, np.array([landmark_loss])))

  else:
    landmark_losses = np.vstack((landmark_losses, np.zeros((1, 33)))) 

  return landmark_losses

og_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='omg_trainer//pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.VIDEO)

prac_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='omg_trainer//pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

og_landmarker = PoseLandmarker.create_from_options(og_options)
prac_landmarker = PoseLandmarker.create_from_options(prac_options)
landmark_losses = np.empty((0, 33), float)

og_video_path = "//Users//rkdbg//Codes//GitHub//rkdbq//ComputerVision//omg_trainer//omg1.mp4"
og_video_capture = cv2.VideoCapture(og_video_path)
playback_speed = 1.75

prac_video_capture = cv2.VideoCapture(0)

frame_width = int(prac_video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(prac_video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(prac_video_capture.get(cv2.CAP_PROP_FPS))
codec = cv2.VideoWriter_fourcc(*'XVID')  # 비디오 코덱 설정 (여기서는 XVID를 사용)

# prac_recorded = cv2.VideoWriter('practice.avi', codec, fps, (frame_width, frame_height))

if not og_video_capture.isOpened():
    print("Error opening video file")
    exit()

while og_video_capture.isOpened():
    timestamp = int(og_video_capture.get(cv2.CAP_PROP_POS_FRAMES))

    og_ret, og_frame = og_video_capture.read()
    prac_ret, prac_frame = prac_video_capture.read()

    if not og_ret:
       break

    og_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=og_frame)
    og_landmarker_result = og_landmarker.detect_for_video(og_image, timestamp)

    prac_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=prac_frame)
    prac_landmarker.detect_async(prac_image, timestamp)

    landmark_losses = calculate_loss(landmark_losses, og_landmarker_result, prac_landmarker_result)

    cv2.imshow('Og', draw_landmarks_on_image(og_image.numpy_view(), og_landmarker_result))
    cv2.imshow('Prac', draw_landmarks_on_image(prac_image.numpy_view(), prac_landmarker_result))

    delay = int(1000 / (og_video_capture.get(cv2.CAP_PROP_FPS) * playback_speed))

    print(np.mean(landmark_losses))

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

og_video_capture.release()
prac_video_capture.release()
cv2.destroyAllWindows()