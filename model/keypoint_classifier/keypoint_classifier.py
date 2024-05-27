import numpy as np
import tensorflow as tf


class KeyPointClassifier(object):
    """
    Класс для классификации ключевых точек с помощью модели TensorFlow Lite.

    Атрибуты:
        interpreter (tf.lite.Interpreter): Интерпретатор модели TensorFlow Lite.
        input_details (dict): Детали входного тензора модели.
        output_details (dict): Детали выходного тензора модели.

    Параметры:
        model_path (str): Путь к файлу модели TensorFlow Lite.
            По умолчанию 'model/keypoint_classifier/keypoint_classifier.tflite'.
        num_threads (int): Количество потоков для вычислений.
            По умолчанию 1.
    """

    def __init__(
            self,
            model_path='model/keypoint_classifier/keypoint_classifier.tflite',
            num_threads=1,
    ):
        """
        Инициализация класса KeyPointClassifier.

        Args:
            model_path (str): Путь к файлу модели TensorFlow Lite.
            num_threads (int): Количество потоков для вычислений.
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
            self,
            landmark_list,
    ):
        """
        Классификация списка ключевых точек с помощью модели TensorFlow Lite.

        Args:
            landmark_list (list or np.ndarray): Список или numpy-массив с координатами ключевых точек.

        Returns:
            int: Индекс класса, соответствующий классифицированному жесту.
        """
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index
