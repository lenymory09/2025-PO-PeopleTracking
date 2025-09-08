import argparse
from tracking5 import track_people
from threading import Thread



def parse_args():
    parser = argparse.ArgumentParser(description="Deep SORT Realtime")
    parser.add_argument("-s", "--source", nargs="+", help="Video Source", default=["./angle2.mp4"])
    return parser.parse_args()


def main():
    args = parse_args()
    threads = []
    for src in args.source:
        t = Thread(target=track_people, args=(src,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
