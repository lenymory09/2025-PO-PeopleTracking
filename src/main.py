import argparse
import cv2
from person_tracker import PersonTracker
from utils import parse_source

def parse_args():
    parser = argparse.ArgumentParser(description="Deep SORT Realtime")
    parser.add_argument("-s", "--source", nargs="+", help="Video Source", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    sources = parse_source(args.source)
    tracker = PersonTracker(sources)

    try:
        tracker.start()
    except KeyboardInterrupt as _:
        print("Fin du programme")
        tracker.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
