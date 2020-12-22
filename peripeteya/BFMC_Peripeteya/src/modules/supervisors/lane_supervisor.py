from threading import Thread

from src.modules.controller.controller import Controller
from src.modules.perception.lane_detection.test_images.image_processing import generate_images
import cv2

from src.modules.supervisors.cnn import load_model, predict
from src.utils.templates.workerprocess import WorkerProcess


class LaneSupervisor:
    def __init__(self):
        self.controller = Controller(5, 5)
        self.controller_brake()
        self.model = load_model()

    def get_command_dictionary(self, image):
        try:
            results = generate_images(image)
            warped = results["warped_thresh"]
            cv2.imwrite("warped.jpg", warped[warped.shape[0]//2:])
            prediction = predict(warped[warped.shape[0]//2:], self.model)
            speed = prediction[1]
            angle = prediction[2]*6
            speed = min(max(speed, -0.22), 0.22)
            self.controller.update_steering_angle(angle)
            self.controller.update_speed(speed)
        except Exception as e:
            self.controller_brake()
            print("Lane Supervisor Exception: ", e)
            raise Exception()
        command = self.controller.get_command_dictionary()
        return command, 0

    def controller_brake(self):
        self.controller.brake()

    def controller_forward(self):
        self.controller.update_speed(0.22)


class LaneDetectionProcess(WorkerProcess):

    def __init__(self, inPs, outPs):
        super(LaneDetectionProcess, self).__init__(inPs, outPs)

        self._image_size = (480, 270, 3)
        self._lane_supervisor = LaneSupervisor()

    def run(self):
        super(LaneDetectionProcess, self).run()

    def _init_threads(self):
        detection_thread = Thread(name="LaneDetection", target=self._detect_lane, args=(self.inPs[0],))
        detection_thread.daemon = True
        self.threads.append(detection_thread)
        pass

    def _detect_lane(self, inP):
        print('Start Detecting Lanes')

        while True:
            offset = 0
            try:
                stamps, image = inP.recv()

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.imwrite("image.jpg", image)
                command, offset = self._lane_supervisor.get_command_dictionary(image)
            except Exception as e:
                print("Lane Detection failed!:", e, "\n")
                self._lane_supervisor.controller_brake()
                pass
            command = self._lane_supervisor.controller.get_command_dictionary()
            for outP in self.outPs:
                outP.send([command, offset])
