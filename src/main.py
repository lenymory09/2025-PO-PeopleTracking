import argparse
import cv2
from person_tracker import PersonTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Deep SORT Realtime")
    parser.add_argument("-s", "--source", nargs="+", help="Video Source", default=["./angle2.mp4"])
    parser.add_argument("--model", help="AI Model", default="osnet")
    return parser.parse_args()


MODELS = [
    'osnet',
    'deepsort'
]


def main():
    args = parse_args()
    sources = args.source if args.source is not None else [0]
    model = args.model
    assert model in MODELS, "Specified model not in models list."
    tracker = PersonTracker(sources, model)

    try:
        tracker.start()
    except KeyboardInterrupt as _:
        print("Fin du programme")
        tracker.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
