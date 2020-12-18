from threading import Thread

from src.modules.controller.controller import Controller
from src.modules.perception.lane_detection.image_processing.image_processing import generate_images
from src.modules.perception.lane_detection.lane_processing.lane_processing import steering_angle, offset_from_center
import time
import picamera
import numpy as np
import cv2

from src.utils.templates.workerprocess import WorkerProcess


class LaneSupervisor:
    def __init__(self):
        self.controller = Controller(30, 30)
        self.controller_brake()

    def get_command_dictionary(self, image):
        offset = 0
        try:
            results = generate_images(image)
            cv2.imwrite("warped.jpg", results["warped"])
            angle = steering_angle(results["warped"])
            print("got angle:", angle)
            offset = offset_from_center(results["warped"])
            print("got offset: ", offset)
            # if np.abs(offset) < 5:
            #     offset = 0
            self.controller.update_steering_angle(angle + offset/10)
            self.controller_forward()
        except Exception:
            self.controller_brake()
            print("Lane Supervisor Exception")
            raise Exception()
        command = self.controller.get_command_dictionary()
        return command, offset

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
