class PiCameraProcess:
    def __init__(self, resolution: tuple):
        self._camera = picamera.PiCamera()
        self._camera.resolution = resolution
        self._camera.start_preview()

    def capture_image(self):
        self._camera.capture("camera_test.jpg", resize=(240, 240))
