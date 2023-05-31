import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

presence_threshold = 0.5

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
def get_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global prac_landmarker_result
    prac_landmarker_result = result

og_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='omg_trainer//pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.VIDEO)

prac_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='omg_trainer//pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=get_result)

og_landmarker = PoseLandmarker.create_from_options(og_options)
prac_landmarker = PoseLandmarker.create_from_options(prac_options)

def frame_with_landmark(frame, timestamp):
   og_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
   og_landmarker_result = og_landmarker.detect_for_video(og_image, timestamp)

   return draw_landmarks_on_image(og_image.numpy_view(), og_landmarker_result), og_landmarker_result

def webcam_frame_with_landmark(frame, timestamp):
    prac_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    prac_landmarker.detect_async(prac_image, timestamp)

    return draw_landmarks_on_image(prac_image.numpy_view(), prac_landmarker_result), prac_landmarker_result

def calculate_landmark_loss(og_landmarker_result, prac_landmarker_result):
   og_len = len(og_landmarker_result.pose_landmarks)
   prac_len = len(prac_landmarker_result.pose_landmarks)
   loss = np.zeros(0)
   if og_len == 0:
      if prac_len == 0:
         loss = 0
      else:
         prac_pose_landmarks = prac_landmarker_result.pose_landmarks[0]
         prac_pose_landmarks = prac_pose_landmarks[:1] + prac_pose_landmarks[7:9] + prac_pose_landmarks[11:]
         for landmark in prac_pose_landmarks:
            landmark_array = np.ones(4)
            if landmark.presence < presence_threshold:
                landmark_array = np.zeros(4)
            loss = np.concatenate((loss, landmark_array**2), axis=0)
         loss = np.mean(loss)
   else:
      if prac_len == 0:
         og_pose_landmarks = og_landmarker_result.pose_landmarks[0]
         og_pose_landmarks = og_pose_landmarks[:1] + og_pose_landmarks[7:9] + og_pose_landmarks[11:]
         for landmark in og_pose_landmarks:
            landmark_array = np.ones(4)
            if landmark.presence < presence_threshold:
                landmark_array = np.zeros(4)
            loss = np.concatenate((loss, landmark_array**2), axis=0)
         loss = np.mean(loss)

      else:
         og_pose_landmarks = og_landmarker_result.pose_landmarks[0]
         prac_pose_landmarks = prac_landmarker_result.pose_landmarks[0]
         og_pose_landmarks = og_pose_landmarks[:1] + og_pose_landmarks[7:9] + og_pose_landmarks[11:]
         prac_pose_landmarks = prac_pose_landmarks[:1] + prac_pose_landmarks[7:9] + prac_pose_landmarks[11:]
         for og_landmark, prac_landmark in zip(og_pose_landmarks, prac_pose_landmarks):
            og_landmark_array = normalize(np.array([og_landmark.x, og_landmark.y, og_landmark.z, og_landmark.visibility]))
            prac_landmark_array = normalize(np.array([prac_landmark.x, prac_landmark.y, prac_landmark.z, prac_landmark.visibility]))
            if og_landmark.presence < presence_threshold:
                og_landmark_array = prac_landmark_array - 1
            if prac_landmark.presence < presence_threshold:
               prac_landmark_array = og_landmark_array - 1
            loss = np.concatenate((loss, ((og_landmark_array) - prac_landmark_array)**2), axis=0)
         loss = np.mean(loss)

   return loss

def normalize(array):
   min_val = np.min(array)
   max_val = np.max(array)
   normalized_arr = (array - min_val) / (max_val - min_val)

   return normalized_arr

import subprocess

def convert_to_mp3(input_file, output_file):
    if os.path.exists(output_file):
       return

    # FFmpeg 명령어 설정
    command = ['ffmpeg', '-i', input_file, '-vn', '-acodec', 'libmp3lame', '-ab', '192k', '-f', 'mp3', output_file]

    # FFmpeg 실행
    subprocess.call(command)

import os

def change_extension(file_path, new_extension):
    # 기존 파일 경로에서 확장자 추출
    base_path, _ = os.path.splitext(file_path)

    # 새로운 확장자를 추가하여 파일 경로 생성
    new_file_path = base_path + new_extension

    return new_file_path    