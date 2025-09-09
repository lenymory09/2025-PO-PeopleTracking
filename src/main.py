import argparse

import cv2

from tracking5 import PersonTracker
from threading import Thread
from typing import List

def parse_args():
    parser = argparse.ArgumentParser(description="Deep SORT Realtime")
    parser.add_argument("-s", "--source", nargs="+", help="Video Source", default=["./angle2.mp4"])
    return parser.parse_args()


def main():
    args = parse_args()
    sources = args.source
    trackers: List[PersonTracker] = []
    known_embeddings = {}
    for source in sources:
        trackers.append(PersonTracker(source))


    while True:
        assigned_id = []
        for i, tracker in enumerate(trackers):
            cv2.imshow(f"stream : {i}", tracker.track_people(known_embeddings, assigned_id))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    for tracker in trackers:
        tracker.cap.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
