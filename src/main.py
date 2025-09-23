import argparse
import cv2
from person_tracker import PersonTracker
from utils import parse_source

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

    try:
        tracker.start()
    except KeyboardInterrupt as _:
        print("Fin du programme")
        tracker.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
