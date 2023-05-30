import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap
from pytube import YouTube
from PyQt5 import QtCore

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 윈도우 설정
        self.setWindowTitle("YouTube Downloader")
        self.setGeometry(200, 200, 400, 300)

        # 위젯 초기화
        self.url_label = QLabel("YouTube URL:")
        self.url_lineedit = QLineEdit()
        self.download_button = QPushButton("Download")
        self.thumbnail_label = QLabel()

        # 레이아웃 초기화
        layout = QVBoxLayout()
        layout.addWidget(self.url_label)
        layout.addWidget(self.url_lineedit)
        layout.addWidget(self.download_button)
        layout.addWidget(self.thumbnail_label)

        # 위젯 설정
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # 버튼 클릭 시 이벤트 연결
        self.download_button.clicked.connect(self.download_video)

    def download_video(self):
        video_url = self.url_lineedit.text()

        try:
            # YouTube 객체 생성
            yt = YouTube(video_url)

            # 가장 높은 품질의 동영상 스트림 선택
            stream = yt.streams.get_highest_resolution()

            # 동영상 다운로드 시작
            print("Downloading...")
            video_path = stream.download(output_path="downloaded_videos")

            # 다운로드한 동영상의 썸네일 표시
            thumbnail_path = yt.thumbnail_url
            self.show_thumbnail(thumbnail_path)

            print("Download completed!")
        except Exception as e:
            print("Error:", str(e))

    def show_thumbnail(self, thumbnail_path):
        # 썸네일 이미지 로드
        pixmap = QPixmap()
        pixmap.loadFromData(thumbnail_path)

        # 이미지 크기 조정
        pixmap = pixmap.scaled(400, 300, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

        # 썸네일 이미지 표시
        self.thumbnail_label.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
