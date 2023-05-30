import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, QFileDialog, QSlider
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from preprocess import frame_with_landmark, webcam_frame_with_landmark

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()

        # 윈도우 설정
        self.setWindowTitle("Video Player")
        self.setGeometry(200, 200, 1600, 600)

        # 비디오 레이블 초기화
        self.video_label = QLabel()
        self.webcam_label = QLabel()

        self.webcam = cv2.VideoCapture(0)
        self.webcam_timer = QTimer()
        self.webcam_timer.timeout.connect(self.update_webcam_frame)
        self.webcam_timer.start(30)
        self.webcam_timestamp = 0

        # 비디오 재생을 위한 타이머 설정
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # 비디오 파일 선택 버튼 초기화
        self.select_button = QPushButton("Select Video")
        self.select_button.clicked.connect(self.select_video)

        # 재생 속도 슬라이더 초기화
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(10)
        self.speed_slider.setValue(5)
        self.speed_slider.valueChanged.connect(self.update_speed)

        # 레이아웃 초기화
        video_layout = QHBoxLayout()
        video_layout.addWidget(self.video_label)
        video_layout.addWidget(self.webcam_label)

        layout = QVBoxLayout()
        layout.addLayout(video_layout)
        layout.addWidget(self.select_button)
        layout.addWidget(self.speed_slider)

        # 위젯 설정
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # 선택한 비디오 파일 경로
        self.video_path = ""

        # 재생 속도
        self.playback_speed = 1.0

    def select_video(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Video Files (*.mp4 *.avi)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            self.video_path = selected_files[0]

            # 선택한 비디오 파일 열기
            self.cap = cv2.VideoCapture(self.video_path)

            # 비디오 재생 시작
            self.timer.start(30)

    def update_speed(self):
        # 재생 속도 슬라이더 값에 따라 재생 속도 설정
        value = self.speed_slider.value()
        self.playback_speed = value / 5.0

    def update_webcam_frame(self):
        # 비디오 프레임 읽기
        self.webcam_timestamp += 1
        ret, cam_frame = self.webcam.read()

        # 프레임 읽기에 실패하면 종료
        if not ret:
            self.webcam_timer.stop()
            return
        
        # 프레임을 QImage로 변환
        cam_frame = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
        cam_frame = cv2.flip(cam_frame, 1)
        landmarked_frame = webcam_frame_with_landmark(cam_frame, self.webcam_timestamp)
        
        height, width, channels = landmarked_frame.shape
        q_image = QImage(
            landmarked_frame.data, width, height, width * channels, QImage.Format_RGB888
        )

        # QImage를 QPixmap으로 변환하여 비디오 레이블에 표시
        pixmap = QPixmap.fromImage(q_image)
        self.webcam_label.setPixmap(pixmap.scaledToWidth(600, Qt.SmoothTransformation))

    def update_frame(self):
        # 비디오 프레임 읽기
        timestamp = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, og_frame = self.cap.read()

        # 프레임 읽기에 실패하면 종료
        if not ret:
            self.timer.stop()
            return

        # 프레임을 QImage로 변환
        og_frame = cv2.cvtColor(og_frame, cv2.COLOR_BGR2RGB)
        landmarked_frame = frame_with_landmark(og_frame, timestamp)

        height, width, channels = landmarked_frame.shape
        q_image = QImage(
            landmarked_frame.data, width, height, width * channels, QImage.Format_RGB888
        )

        # QImage를 QPixmap으로 변환하여 비디오 레이블에 표시
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap.scaledToWidth(600, Qt.SmoothTransformation))

        # 재생 속도 적용
        delay = int(1000 / (self.cap.get(cv2.CAP_PROP_FPS) * self.playback_speed))

        # 딜레이 후 타이머 재실행
        self.timer.start(delay)


if __name__ == "__main__":
    app = QApplication([])
    player = VideoPlayer()
    player.show()
    app.exec_()
