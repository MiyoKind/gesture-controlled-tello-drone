from controllers.interfaces.velocity_controller_interface import VelocityControllerInterface
from constants import MAX_VELOCITY, SMOOTHING_FACTOR, VELOCITY_CONTROLLER_THRESHOLD


class VelocityController(VelocityControllerInterface):
    """
        Класс для управления скоростями дрона.

        Attributes:
            max_velocity (int): Максимальная скорость.
            smoothing_factor (float): Коэффициент сглаживания скорости.
            forw_back_velocity (int): Текущая скорость движения вперед/назад.
            up_down_velocity (int): Текущая скорость движения вверх/вниз.
            left_right_velocity (int): Текущая скорость движения влево/вправо.
            yaw_velocity (int): Текущая скорость вращения по оси yaw.
            smoothed_forw_back_velocity (int): Сглаженная скорость движения вперед/назад.
            smoothed_up_down_velocity (int): Сглаженная скорость движения вверх/вниз.
            smoothed_left_right_velocity (int): Сглаженная скорость движения влево/вправо.
            smoothed_yaw_velocity (int): Сглаженная скорость вращения по оси yaw.
            velocity_threshold (int): Порог для отправки команд управления.
            last_velocities (list): Список последних отправленных скоростей.
    """
    def __init__(self, max_velocity=MAX_VELOCITY, smoothing_factor=SMOOTHING_FACTOR):
        self.max_velocity = max_velocity
        self.smoothing_factor = smoothing_factor

        self.forw_back_velocity = 0
        self.up_down_velocity = 0
        self.left_right_velocity = 0
        self.yaw_velocity = 0

        self.smoothed_forw_back_velocity = 0
        self.smoothed_up_down_velocity = 0
        self.smoothed_left_right_velocity = 0
        self.smoothed_yaw_velocity = 0

        self.velocity_threshold = VELOCITY_CONTROLLER_THRESHOLD

        self.last_velocities = [0, 0, 0, 0]

    def reset_velocities(self):
        self.forw_back_velocity = 0
        self.up_down_velocity = 0
        self.left_right_velocity = 0
        self.yaw_velocity = 0
        self.smoothed_forw_back_velocity = 0
        self.smoothed_up_down_velocity = 0
        self.smoothed_left_right_velocity = 0
        self.smoothed_yaw_velocity = 0

    def update_velocity(self, velocity_name, direction):
        velocity = getattr(self, velocity_name)
        smoothed_velocity = getattr(self, 'smoothed_' + velocity_name)

        # Постоянная величина для увеличения скорости
        velocity_step = 1  # Можно настроить это значение

        if direction > 0:
            velocity = min(velocity + velocity_step, self.max_velocity)
        else:
            velocity = max(velocity - velocity_step, -self.max_velocity)

        # Экспоненциальное сглаживание
        smoothed_velocity = int(self.smoothing_factor * velocity + (1 - self.smoothing_factor) * smoothed_velocity)

        setattr(self, velocity_name, velocity)
        setattr(self, 'smoothed_' + velocity_name, smoothed_velocity)

    def get_velocities(self):
        return [
            self.smoothed_left_right_velocity,
            self.smoothed_forw_back_velocity,
            self.smoothed_up_down_velocity,
            self.smoothed_yaw_velocity,
        ]
