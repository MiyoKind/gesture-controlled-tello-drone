from constants import LANDING_GESTURE_HOLD_TIME
from constants import FLIP_GESTURE_HOLD_TIME
from controllers.interfaces.tello_controller_interface import TelloControllerInterface
from controllers.interfaces.velocity_controller_interface import VelocityControllerInterface


class GestureController:
    """
        Класс для управления дроном с помощью жестов.

        Attributes:
            tello_controller (TelloControllerInterface): Контроллер для управления дроном.
            velocity_controller (VelocityControllerInterface): Контроллер для управления скоростями дрона.
            in_flight (bool): Флаг, указывающий, находится ли дрон в полете.
            takeoff_handled (bool): Флаг, указывающий, был ли обработан жест взлета.
            _is_landing (bool): Флаг, указывающий, выполняется ли посадка дрона.
            landing_gesture_timer (int): Таймер для удержания жеста посадки.
            flip_gesture_timer (int): Таймер для удержания жеста переворота.
    """
    def __init__(self, tello_controller: TelloControllerInterface, velocity_controller: VelocityControllerInterface):
        """
        Инициализирует объект GestureController.

        Args:
            tello_controller (TelloControllerInterface): Контроллер для управления дроном.
            velocity_controller (VelocityControllerInterface): Контроллер для управления скоростями дрона.
        """
        self.tello_controller = tello_controller
        self.velocity_controller = velocity_controller
        self.in_flight = False
        self.takeoff_handled = False
        self._is_landing = False
        self.landing_gesture_timer = 0
        self.flip_gesture_timer = 0

    def handle_gesture(self, gesture_id):
        """
        Обрабатывает жест в зависимости от его идентификатора.

        Args:
            gesture_id (int): Идентификатор жеста.
        """
        print("GESTURE: ", gesture_id)
        if self._is_landing:
            return

        is_stopped = False

        if not self.in_flight and not self.takeoff_handled:
            if gesture_id == 0:
                self.handle_takeoff()
            else:
                return

        if gesture_id == 2:
            self.handle_stop()
            is_stopped = True
        elif self.in_flight:
            gesture_actions = {
                1: self.handle_land,
                3: lambda: self.velocity_controller.update_velocity('up_down_velocity', 1),
                4: lambda: self.velocity_controller.update_velocity('up_down_velocity', -1),
                5: lambda: self.velocity_controller.update_velocity('forw_back_velocity', -1),
                6: lambda: self.velocity_controller.update_velocity('forw_back_velocity', 1),
                7: lambda: self.velocity_controller.update_velocity('left_right_velocity', 1),
                8: lambda: self.velocity_controller.update_velocity('left_right_velocity', -1),
                9: lambda: self.velocity_controller.update_velocity('yaw_velocity', 1),
                10: lambda: self.velocity_controller.update_velocity('yaw_velocity', -1),
                11: self.handle_flip('f', gesture_id),
                12: self.handle_flip('b', gesture_id),
                13: self.handle_flip('l', gesture_id),
                14: self.handle_flip('r', gesture_id)
            }

            if gesture_id in gesture_actions:
                gesture_actions[gesture_id]()
                self.flip_gesture_timer = 0
            elif gesture_id == -1:
                self.velocity_controller.reset_velocities()

        if not is_stopped:
            self.send_control_command()

    def handle_takeoff(self):
        """
        Обрабатывает жест взлета.
        """
        self.tello_controller.takeoff()
        if self.tello_controller.tello.is_flying:
            self.in_flight = True
            self.takeoff_handled = True

    def handle_land(self):
        """
        Обрабатывает жест посадки.
        """
        if self.landing_gesture_timer >= LANDING_GESTURE_HOLD_TIME:
            self.tello_controller.land()
            self.in_flight = False
            self.takeoff_handled = False
            self._is_landing = False
            self.landing_gesture_timer = 0
        else:
            self.landing_gesture_timer += 1

    def handle_stop(self):
        """
        Обрабатывает жест остановки.
        """
        self.send_stop_command()  # Отправляем команду остановки напрямую в TelloController
        self.velocity_controller.reset_velocities()
        self.landing_gesture_timer = 0  # Сбрасываем таймер жеста посадки
        self.flip_gesture_timer = 0

    def send_stop_command(self):
        """
        Отправляет команду остановки дрона.
        """
        self.tello_controller.send_control_command(0, 0, 0, 0)

    def handle_flip(self, direction, gesture_id):
        """
        Обрабатывает жест переворота.

        Args:
            direction (str): Направление переворота ('f', 'b', 'l', 'r').
            gesture_id (int): Идентификатор жеста.
        """
        if gesture_id != -1 and (11 <= gesture_id <= 14) and (self.flip_gesture_timer >= FLIP_GESTURE_HOLD_TIME):
            self.tello_controller.flip(direction)
            self.flip_gesture_timer = 0
        else:
            self.flip_gesture_timer += 1

    def send_control_command(self):
        """
        Отправляет команду управления дроном с текущими скоростями.
        """
        velocities = self.velocity_controller.get_velocities()
        self.tello_controller.send_control_command(*velocities)
