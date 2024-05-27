import configargparse


def get_args():
    """
    Парсинг аргументов командной строки и конфигурационных файлов.

    Returns:
    argparse.Namespace: Объект с разобранными аргументами.
    """
    print('## Reading configuration ##')
    parser = configargparse.ArgParser(default_config_files=['config.txt'])

    parser.add_argument('-c', '--my-config', required=False, is_config_file=True, help='config file path')
    parser.add_argument("--device", type=int)
    parser.add_argument("--width", help='cap width', type=int)
    parser.add_argument("--height", help='cap height', type=int)
    parser.add_argument('--use_static_image_mode', action='store_true', help='True if running on photos')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=float)
    parser.add_argument("--buffer_len",
                        help='Length of gesture buffer',
                        type=int)

    return parser.parse_args()
