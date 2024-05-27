import copy
import csv

import cv2 as cv
import mediapipe as mp

from image_processing.gesture_drawer import GestureDrawer
from image_processing.image_processor import ImageProcessor
from model import KeyPointClassifier
from constants import USE_BRECT


class GestureRecognition:
    """
        Класс для распознавания жестов рук.

        Attributes:
            use_static_image_mode (bool): Режим обработки статических изображений.
            min_detection_confidence (float): Минимальный порог уверенности для детекции.
            min_tracking_confidence (float): Минимальный порог уверенности для отслеживания.
            hands (mp.solutions.hands.Hands): Модель для детекции рук.
            keypoint_classifier (KeyPointClassifier): Классификатор ключевых точек.
            keypoint_classifier_labels (list): Метки для классификатора ключевых точек.
            gesture_timer (int): Таймер удержания жеста.
            prev_gesture_id (int): Предыдущий идентификатор жеста.
        """
    def __init__(self, use_static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        """
        Инициализирует объект GestureRecognition.
        """
        self.use_static_image_mode = use_static_image_mode
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Загрузка моделей
        self.hands, self.keypoint_classifier, self.keypoint_classifier_labels = self.load_model()

        # Таймер удержания жеста
        self.gesture_timer = 0
        self.prev_gesture_id = -1

    def on_gesture_changed(self, gesture_id):
        """
        Обработка изменения жеста.
        """
        if gesture_id != self.prev_gesture_id:
            self.gesture_timer = 0
            self.prev_gesture_id = gesture_id
        else:
            self.gesture_timer += 1

    def load_model(self):
        """
        Загрузка моделей для распознавания жестов.
        """
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=self.use_static_image_mode,
            max_num_hands=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        keypoint_classifier = KeyPointClassifier()

        # Чтение меток ###########################################################
        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                  encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [
                row[0] for row in keypoint_classifier_labels
            ]

        return hands, keypoint_classifier, keypoint_classifier_labels

    def recognize(self, image):
        """
        Распознавание жеста на изображении.

        Args:
            image (np.ndarray): Входное изображение.

        Returns:
            tuple: Tuple containing:
            detector_image (np.ndarray): Изображение с отображением результатов распознавания.
            gesture_id (int): Идентификатор распознанного жеста.
            gesture_timer (int): Таймер удержания жеста.
        """
        use_brect = USE_BRECT

        image = cv.flip(image, 1)
        detector_image = copy.deepcopy(image)

        # Сохранение идентификатора жеста
        gesture_id = -1

        # Реализация детектора #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if (hasattr(results, 'multi_hand_landmarks') and hasattr(results, 'multi_handedness')
                and results.multi_hand_landmarks):
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Вычисление ограничивающего прямоугольника
                brect = ImageProcessor.calc_bounding_rect(detector_image, hand_landmarks)
                # Вычисление ключевых точек
                landmark_list = ImageProcessor.calc_landmark_list(detector_image, hand_landmarks)

                # Нормализация координат ключевых точек
                pre_processed_landmark_list = ImageProcessor.pre_process_landmark(landmark_list)

                # Классификация жеста
                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)

                # Отрисовка результата
                detector_image = GestureDrawer.draw_bounding_rect(use_brect, detector_image, brect)
                detector_image = GestureDrawer.draw_landmarks(detector_image, landmark_list)
                detector_image = GestureDrawer.draw_info_text(
                    detector_image,
                    brect,
                    handedness,
                    self.keypoint_classifier_labels[hand_sign_id]
                )

                # Сохранение жеста
                gesture_id = hand_sign_id

                # Обновление таймера удержания жеста
                self.on_gesture_changed(gesture_id)

        return detector_image, gesture_id, self.gesture_timer
