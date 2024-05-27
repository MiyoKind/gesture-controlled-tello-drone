from abc import ABC, abstractmethod


class VelocityControllerInterface(ABC):
    """
    Абстрактный базовый класс для контроллеров скоростей дрона.
    """
    @abstractmethod
    def reset_velocities(self):
        """
        Сбрасывает все скорости дрона до нуля.
        """
        pass

    @abstractmethod
    def update_velocity(self, velocity_name, direction):
        """
        Обновляет скорость движения дрона в указанном направлении.

        Args:
            velocity_name (str): Имя скорости ('left_right', 'forw_back', 'up_down', 'yaw').
            direction (int): Направление скорости (-1 или 1).
        """
        pass

    @abstractmethod
    def get_velocities(self):
        """
        Возвращает текущие скорости движения дрона.

        Returns:
            tuple: Кортеж со скоростями (left_right_velocity, forw_back_velocity, up_down_velocity, yaw_velocity).
        """
        pass
