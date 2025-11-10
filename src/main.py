import argparse
import sys
import yaml
from PySide6 import QtWidgets

from gui import GUIApp

def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(description="Deep SORT Realtime")
    parser.add_argument("-s", "--source", nargs="+", help="Video Source", required=True)
    return parser.parse_args()


def main():
    # args = parse_args()
    app = QtWidgets.QApplication([])
    app.setApplicationName("Person Tracker")
    main_config = load_config()
    gui_app = GUIApp(main_config)
    gui_app.show()
    gui_app.start_processing()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
