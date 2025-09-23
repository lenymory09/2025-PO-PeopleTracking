import argparse
import sys

import PySide6
import cv2
from PySide6 import QtWidgets

from person_tracker import PersonTracker
from utils import parse_source
from gui import GUIApplication


def parse_args():
    parser = argparse.ArgumentParser(description="Deep SORT Realtime")
    parser.add_argument("-s", "--source", nargs="+", help="Video Source", required=True)
    parser.add_argument("--model", help="AI Model", default="osnet")
    return parser.parse_args()


MODELS = [
    'osnet',
    'deepsort'
]


def main():
    args = parse_args()
    sources = parse_source(args.source)
    model = args.model
    assert model in MODELS, "Specified model not in models list."
    tracker = PersonTracker(sources, model)
    app = QtWidgets.QApplication([])
    app.setApplicationName("Person Tracker")
    gui_app = GUIApplication(tracker)

    gui_app.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
