from abc import ABC, abstractmethod
from djitellopy import Tello


class TelloControllerInterface(ABC):
    """
    Абстрактный базовый класс для контроллеров дрона Tello.
    """
    @abstractmethod
    def __init__(self, tello: Tello):
        """
        Инициализирует объект контроллера.

        Args:
            tello (Tello): Объект дрона Tello.
        """
        self.tello = tello
        pass

    @abstractmethod
    def takeoff(self):
        """
        Выполняет взлет дрона.
        """
        pass

    @abstractmethod
    def land(self):
        """
        Выполняет посадку дрона.
        """
        pass

    @abstractmethod
    def flip(self, direction):
        """
        Выполняет переворот дрона в указанном направлении.

        Args:
            direction (str): Направление переворота ('l', 'r', 'f', 'b').
        """
        pass

    @abstractmethod
    def send_control_command(self, left_right_velocity, forw_back_velocity, up_down_velocity, yaw_velocity):
        """
        Отправляет команду управления дроном.

        Args:
            left_right_velocity (int): Скорость движения влево/вправо (-100 - 100).
            forw_back_velocity (int): Скорость движения вперед/назад (-100 - 100).
            up_down_velocity (int): Скорость движения вверх/вниз (-100 - 100).
            yaw_velocity (int): Скорость вращения по оси yaw (-100 - 100).
        """
        pass
