from djitellopy import Tello
from controllers.interfaces.tello_controller_interface import TelloControllerInterface
from math import sqrt, pow


class TelloController(TelloControllerInterface):
    """
    Класс для управления дроном Tello.

    Attributes:
        tello (Tello): Объект дрона Tello.
    """
    def __init__(self, tello: Tello):
        self.tello = tello

    def takeoff(self):
        self.tello.takeoff()

    def land(self):
        self.tello.land()

    def flip(self, direction):
        self.tello.flip(direction)

    def get_battery(self):
        return self.tello.get_battery()

    def get_height(self):
        return self.tello.get_height()

    def get_speed(self):
        return sqrt(pow(self.tello.get_speed_x(), 2) + pow(self.tello.get_speed_y(), 2))

    def send_control_command(self, left_right_velocity, forw_back_velocity, up_down_velocity, yaw_velocity):
        self.tello.send_rc_control(
            left_right_velocity,
            forw_back_velocity,
            up_down_velocity,
            yaw_velocity,
        )
