import time
from threading import Thread, Event
from pywifi import PyWiFi, const, Profile
from constants import TELLO_SSID, WIFI_SIGNAL_THRESHOLD


class WiFiManager:
    """
    Класс для мониторинга соединения к WiFi сети дрона Tello
    """
    def __init__(self, tello_ssid=TELLO_SSID, signal_threshold=WIFI_SIGNAL_THRESHOLD):
        """
        Инициализирует объект WiFiManager.

        Args:
            tello_ssid (str, optional): SSID сети Tello. По умолчанию используется константа TELLO_SSID.
            signal_threshold (int, optional): Пороговое значение уровня сигнала Wi-Fi. По умолчанию используется константа WIFI_SIGNAL_THRESHOLD.
        """
        self.tello_ssid = tello_ssid
        self.signal_threshold = signal_threshold
        self.wifi = PyWiFi()
        self.iface = self.wifi.interfaces()[0]
        self.connected = False
        self.monitor_event = Event()
        self.monitor_thread = Thread(target=self.monitor_connection)

    def __enter__(self):
        """
        Контекстный менеджер для запуска потока мониторинга подключения.

        Returns:
            WiFiManager: Объект WiFiManager.
        """
        self.monitor_thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Контекстный менеджер для остановки мониторинга подключения и отключения от сети.

        Args:
            exc_type (Exception type, optional): Тип возникшего исключения.
            exc_value (Exception, optional): Объект исключения.
            traceback (Traceback, optional): Трассировка стека.
        """
        self.stop_monitoring()
        self.disconnect()

    def connect(self):
        """
        Подключается к сети Tello Wi-Fi.

        Returns:
            bool: True, если подключение успешно, иначе False.
        """
        if self.is_connected():
            print("Already connected to Tello WiFi.")
            self.connected = True
            return True

        self.iface.disconnect()
        time.sleep(5)

        while True:
            self.iface.scan()
            time.sleep(5)
            networks = self.iface.scan_results()
            for network in networks:
                if network.ssid == self.tello_ssid:
                    try:
                        profile = Profile()
                        profile.ssid = self.tello_ssid
                        profile.auth = const.AUTH_ALG_OPEN
                        profile.akm.append(const.AKM_TYPE_NONE)
                        profile.cipher = const.CIPHER_TYPE_NONE
                        self.iface.remove_all_network_profiles()
                        self.iface.add_network_profile(profile)

                        self.iface.connect(profile)
                        time.sleep(10)

                        if self.is_connected():
                            print("Connected to Tello WiFi.")
                            self.connected = True
                            return True
                    except Exception as e:
                        print(f"Error connecting to Tello WiFi: {e}")
                        self.iface.disconnect()
                        time.sleep(5)
                        break

            print("Tello WiFi not found. Retrying...")
            time.sleep(5)

    def is_connected(self):
        """
        Проверяет, подключен ли объект к сети Tello Wi-Fi.

        Returns:
            bool: True, если подключен к сети Tello Wi-Fi, иначе False.
        """
        if self.iface.status() == const.IFACE_CONNECTED:
            current_profile = self.iface.network_profiles()[0] if self.iface.network_profiles() else None
            return current_profile and current_profile.ssid == self.tello_ssid
        return False

    def disconnect(self):
        """
        Отключается от сети Tello Wi-Fi.
        """
        self.iface.disconnect()
        self.connected = False
        print("Disconnected from Tello WiFi.")

    def monitor_connection(self):
        """
        Мониторит подключение к сети Tello Wi-Fi. При потере подключения пытается переподключиться.
        """
        while True:
            self.monitor_event.wait(timeout=5)
            if not self.is_connected():
                print("Lost connection to Tello WiFi. Attempting to reconnect...")
                self.connect()
            elif self.connected:
                signal_level = self.iface.scan_results()[0].signal
                if signal_level < self.signal_threshold:
                    print(f"Warning: Weak signal strength ({signal_level} dBm). You may be out of range.")
            else:
                print("Reconnected to Tello WiFi.")
                self.connected = True

    def start_monitoring(self):
        """
        Запускает мониторинг подключения к сети Tello Wi-Fi.
        """
        self.monitor_event.set()

    def stop_monitoring(self):
        """
        Останавливает мониторинг подключения к сети Tello Wi-Fi.
        """
        self.monitor_event.clear()
