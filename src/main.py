import argparse
import cv2
from tracking5 import PersonTracker
from typing import List


def parse_args():
    parser = argparse.ArgumentParser(description="Deep SORT Realtime")
    parser.add_argument("-s", "--source", nargs="+", help="Video Source", default=["./angle2.mp4"])
    return parser.parse_args()


def main():
    args = parse_args()
    sources = args.source
    tracker = PersonTracker(sources)

    try:
        for _ in range(500):
            for i, frame in enumerate(tracker.get_frames()):
                cv2.imshow(f"Stream : {i}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt("Keyboard Interrupt raise")
    except KeyboardInterrupt as _:
        print("Fin du programme")
        tracker.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
