from peripeteya.BFMC_Peripeteya.src.modules.controller.controller import Controller
from src.modules.perception.lane_detection.image_processing.image_processing import generate_images
from src.modules.perception.lane_detection.lane_processing.lane_processing import steering_angle, offset_from_center
import time
import picamera
import numpy as np
import cv2


class LaneSupervisor:
	def __init__(self):
		self.controller = Controller(5, 5)
		self.controller_brake()
		self._camera = picamera.PiCamera()

	def get_command_dictionary(self):
		print("Mini supervisor: getting command dictionary")
		self._camera.start_preview()
		print(self._camera.resolution)
		print(self._camera.contrast)
		print(self._camera.brightness)

		if self._camera is None:
			self.controller_brake()
			return self.controller.get_command_dictionary()
		try:
			print("started preview")
			self._camera.capture("src/data/preview.jpg")
			print("image captured")
			image = cv2.imread("src/data/preview.jpg")
			print("image read with cv2")
			cv2.imwrite("src/data/result.jpg", image)
			print("image written")
			results = generate_images(image)
			print("results generated")
			cv2.imwrite("src/data/warped.jpg", results["warped"])
			print("warped is written")
			angle = steering_angle(results["warped"])
			print("got angle")
			offset = offset_from_center(results["warped"])
			print("got offset")
			if np.abs(offset) < 5:
				offset = 0
			self.controller.update_steering_angle(angle+offset/2)
			self.controller_forward()
		except Exception:
			self.controller_brake()
			print("exception")
			raise Exception()
		finally:
			self._camera.stop_preview()
			self._camera.close()
		command = self.controller.get_command_dictionary()
		return command

	def controller_brake(self):
		self.controller.brake()

	def controller_forward(self):
		self.controller.update_speed(0.22)


super = LaneSupervisor()
super.controller_forward()
print(super.get_command_dictionary())
