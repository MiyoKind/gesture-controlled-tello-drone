import time
import cv2 as cv
import logging

from queue import Queue
from threading import Thread, Event
from djitellopy import Tello
from controllers.gesture_controller import GestureController
from image_processing.gesture_recognition import GestureRecognition
from controllers.tello_controller import TelloController
from controllers.velocity_controller import VelocityController
from image_processing.gesture_drawer import GestureDrawer
from utils.argument_parser import get_args
from utils.wifi_manager import WiFiManager
from constants import LOW_BATTERY

# Настройка логгирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    """
    Главная функция, которая запускает основной цикл программы.
    """
    stop_event = Event()
    tello = None
    tello_thread = None
    webcam = None
    wifi_manager = None

    try:
        args = get_args()
        wifi_manager = connect_to_tello_wifi()

        tello, cap = initialize_tello_and_camera()

        tello_controller = TelloController(tello)
        velocity_controller = VelocityController()
        gesture_controller = GestureController(tello_controller, velocity_controller)
        gesture_detector = GestureRecognition(args.use_static_image_mode, args.min_detection_confidence,
                                              args.min_tracking_confidence)

        tello_event = Event()
        gesture_queue = Queue()

        tello_thread = Thread(target=tello_control_thread, args=(gesture_queue, tello_event,
                                                                 gesture_controller, stop_event))
        tello_thread.start()

        setup_opencv_windows()
        webcam = initialize_webcam()

        run_gesture_recognition_loop(cap, webcam, gesture_detector, gesture_queue, tello_event, stop_event,
                                     tello_controller, wifi_manager)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        if webcam is not None:
            webcam.release()
        if wifi_manager is not None:
            wifi_manager.stop_monitoring()
            wifi_manager.disconnect()
        cv.destroyAllWindows()
        cleanup(tello, tello_thread, webcam, wifi_manager, stop_event)


def connect_to_tello_wifi():
    """
    Подключается к сети Tello Wi-Fi и возвращает объект WiFiManager.

    Returns:
        WiFiManager: Объект WiFiManager.
    """
    with WiFiManager() as wifi_manager:
        while not wifi_manager.connect():
            print("Waiting for connection to Tello WiFi...")
            time.sleep(5)
        wifi_manager.start_monitoring()
    return wifi_manager


def initialize_tello_and_camera():
    """
    Инициализирует объект Tello и захват видео с камеры дрона.

    Returns:
        Tello: Объект Tello.
        frame_read: Объект для чтения кадров с камеры дрона.
    """
    tello = Tello()
    tello.connect()
    tello.streamon()
    cap = tello.get_frame_read()
    return tello, cap


def setup_opencv_windows():
    """
    Создает окна OpenCV для отображения изображений с камеры дрона и веб-камеры.
    """
    cv.namedWindow('Drone Camera Feed')
    cv.namedWindow('Webcam Feed')


def initialize_webcam():
    """
    Инициализирует веб-камеру.

    Returns:
        VideoCapture: Объект веб-камеры или None, если не удалось открыть веб-камеру.
    """
    webcam = cv.VideoCapture(0)
    if not webcam.isOpened():
        print("Error: Couldn't open webcam")
        return None
    return webcam


def tello_control_thread(gesture_queue, tello_event, gesture_controller, stop_event):
    """
        Поток для обработки жестов и управления дроном.

        Args:
            gesture_queue (Queue): Очередь для получения жестов.
            tello_event (Event): Событие для синхронизации с главным циклом.
            gesture_controller (GestureController): Контроллер для обработки жестов.
            stop_event (Event): Событие для остановки потока.
    """
    while not stop_event.is_set():
        try:
            gesture = gesture_queue.get()
            if gesture is None:
                continue

            gesture_controller.handle_gesture(gesture)
            tello_event.set()
        except Exception as e:
            print(f"Error occurred in gesture control: {e}")


def run_gesture_recognition_loop(cap, webcam, gesture_detector, gesture_queue, tello_event, stop_event,
                                 tello_controller, wifi_manager):
    """
        Главный цикл распознавания жестов и управления дроном.

        Args:
            cap (frame_read): Объект для чтения кадров с камеры дрона.
            webcam (VideoCapture): Объект веб-камеры.
            gesture_detector (GestureRecognition): Детектор жестов.
            gesture_queue (Queue): Очередь для передачи распознанных жестов.
            tello_event (Event): Событие для синхронизации с потоком обработки жестов.
            stop_event (Event): Событие для остановки цикла.
            tello_controller (TelloController): Контроллер для управления дроном.
            wifi_manager (WiFiManager): Менеджер Wi-Fi подключения.
    """
    last_gesture_time = time.time()
    while not stop_event.is_set():
        webcam_image = get_webcam_frame(webcam)
        drone_image = cap.frame

        if drone_image is None:
            print("No frame received from drone")
            continue

        display_drone_camera_feed(drone_image, tello_controller, wifi_manager, stop_event)
        detector_image, gesture_id, _ = gesture_detector.recognize(webcam_image)
        detector_image = GestureDrawer.draw_info(detector_image)
        display_webcam_feed(detector_image)
        gesture_queue.put(gesture_id)

        if time.time() - last_gesture_time > 0.1:
            tello_event.wait(timeout=0.1)
            tello_event.clear()
            last_gesture_time = time.time()

        check_exit_key(stop_event)


def get_webcam_frame(webcam):
    """
        Получает кадр с веб-камеры.

        Args:
            webcam (VideoCapture): Объект веб-камеры.

        Returns:
            ndarray: Изображение с веб-камеры.
    """
    ret, webcam_image = webcam.read()
    if not ret:
        print("Error: Couldn't read frame from webcam")
    return webcam_image


def display_drone_camera_feed(drone_image, tello_controller, wifi_manager, stop_event):
    """
        Отображает изображение с камеры дрона и информацию о батарее, высоте, скорости и уровне сигнала Wi-Fi.

        Args:
            drone_image (ndarray): Изображение с камеры дрона.
            tello_controller (TelloController): Контроллер для управления дроном.
            wifi_manager (WiFiManager): Менеджер Wi-Fi подключения.
            stop_event (Event): Событие для остановки цикла.
    """
    battery = tello_controller.get_battery()
    height = tello_controller.get_height()
    speed = tello_controller.get_speed()
    signal_level = wifi_manager.iface.scan_results()[0].signal if wifi_manager.is_connected() else -100

    # Отображение информации о батарее, высоте и скорости
    cv.putText(drone_image, f"Battery: {battery}%", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
    cv.putText(drone_image, f"Height: {height:.2f} cm", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
    cv.putText(drone_image, f"Speed: {speed:.2f} cm/s", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)

    # Отображение предупреждения о слабом сигнале
    if signal_level < wifi_manager.signal_threshold:
        cv.putText(drone_image, "WEAK SIGNAL", (drone_image.shape[1] // 2 - 150, drone_image.shape[0] // 2),
                   cv.FONT_HERSHEY_SIMPLEX,
                   2, (0, 0, 255), 4, cv.LINE_AA)
        cv.rectangle(drone_image, (0, 0), (drone_image.shape[1], drone_image.shape[0]), (0, 0, 255), 10)

    # Отображение изображения с камеры дрона
    cv.imshow('Drone Camera Feed', drone_image)

    # Проверка низкого заряда батареи
    if battery <= LOW_BATTERY:
        print("Low battery!")
        stop_event.set()

    cv.waitKey(1)


def display_webcam_feed(debug_image):
    """
        Отображает изображение с веб-камеры и информацию о распознанных жестах.

        Args:
            debug_image (ndarray): Изображение с веб-камеры с отображаемой информацией о распознанных жестах.
    """
    cv.imshow('Webcam Feed', debug_image)
    cv.waitKey(1)


def check_exit_key(stop_event):
    """
        Проверяет нажатие клавиши для выхода из программы.

        Args:
            stop_event (Event): Событие для остановки программы.
    """
    key = cv.waitKey(1) & 0xFF
    if key == 27:
        stop_event.set()


def cleanup(tello, tello_thread, webcam, wifi_manager, stop_event):
    """
        Выполняет очистку ресурсов перед выходом из программы.

        Args:
            tello (Tello): Объект Tello.
            tello_thread (Thread): Поток обработки жестов.
            webcam (VideoCapture): Объект веб-камеры.
            wifi_manager (WiFiManager): Менеджер Wi-Fi подключения.
            stop_event (Event): Событие для остановки программы.
    """
    stop_event.set()
    if tello is not None:
        tello.end()
    if tello_thread is not None:
        tello_thread.join()
    if webcam is not None:
        webcam.release()
    if wifi_manager is not None:
        wifi_manager.stop_monitoring()
        wifi_manager.disconnect()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
