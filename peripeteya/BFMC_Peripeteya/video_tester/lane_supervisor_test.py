import math
import cv2

from src.modules.perception.lane_detection.image_processing.image_processing import generate_images, annotate_image, \
    proper_size
from src.modules.supervisors.lane_supervisor import LaneSupervisor

video = cv2.VideoCapture("video_test.mp4")
supervisor = LaneSupervisor()
while video.isOpened():
    ret, image = video.read()
    if not ret:
        break
    image = proper_size(image)
    command, offset = supervisor.get_command_dictionary(image)
    results = generate_images(image)
    warped = results['blurred']
    angle = command["steerAngle"]
    message = "Left" if angle < 0 else "Right"
    if math.fabs(angle) < 3:
        message = "Forward"
    annotate_image(warped, message + str(angle) + " " + str(offset), (warped.shape[1]//2-45, warped.shape[0]-10))
    warped = cv2.resize(warped, (warped.shape[1]*3, warped.shape[0]*3))
    cv2.imshow("Result", warped)

    print(command)
    cv2.waitKey(10)
