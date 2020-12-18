class Controller:
    def __init__(self, speed_smooth_amount, steer_angle_smooth_amount):
        self._brake = False
        self._speed = 0
        self._steering_angle = 0
        self._speed_smooth = speed_smooth_amount
        self._steer_smooth = steer_angle_smooth_amount
        self._speeds_queue = []
        self._steering_queue = []

    def update_speed(self, new_speed):
        self._speeds_queue.append(new_speed)
        if len(self._speeds_queue) > self._speed_smooth:
            self._speeds_queue.pop(0)
        self._speed = self.get_speed()

    def get_speed(self):
        if len(self._speeds_queue) == 0:
            return 0.
        speed = sum(self._speeds_queue) / len(self._speeds_queue)
        return float("%.2f" % speed)

    def update_steering_angle(self, new_angle):
        self._steering_queue.append(new_angle)
        if len(self._steering_queue) > self._steer_smooth:
            self._steering_queue.pop(0)
        self._steering_angle = self.get_steering()

    def get_steering(self):
        if len(self._steering_queue) == 0:
            return 0.
        angle = sum(self._steering_queue) / len(self._steering_queue)
        return float("%.2f" % angle)

    def brake(self):
        self._brake = True
        self._speeds_queue.clear()
        self._speed = 0

    def reset_brake(self):
        self._brake = False

    def get_command_dictionary(self):
        # {'action': 'MCTL', 'speed': 1.0, 'steerAngle': 0.0}
        command = {
            "action": "MCTL",
            "steerAngle": self.get_steering()
        }
        if self._brake:
            command["action"] = "BRAK"
            self.reset_brake()
            return command
        command["speed"] = self.get_speed()
        return command
