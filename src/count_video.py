import argparse
import json
import cv2

from perception import Perception
from interaction import InteractionLogic


def _print_final(counts, as_json=False):
    if as_json:
        print(json.dumps(counts, ensure_ascii=False), flush=True)
        return
    if counts:
        for k, v in counts.items():
            suff = "" if k.endswith("s") else "s"
            print(f"{k}{suff}: {v}", flush=True)
    else:
        print("No objects counted.", flush=True)


def parse_args():
    p = argparse.ArgumentParser("count_video")
    p.add_argument("video", help="Path to video file")
    p.add_argument("--show", action="store_true", help="Show live window with overlays")

    p.add_argument("--labels", default="data/labels.txt")
    p.add_argument("--model", default="yolov8s.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--json", action="store_true")
    p.add_argument("--mode", choices=["in", "net"], default="in")
    p.add_argument("--zone", type=str, default="FULL")

    args = p.parse_args()

    if not args.json:
        args.json = True

    return args


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}\n")

    ok, frame0 = cap.read()
    if not ok:
        raise SystemExit("Empty video")

    frame0 = cv2.rotate(frame0, cv2.ROTATE_90_CLOCKWISE)
    H, W = frame0.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    perception = Perception(
        model_path=args.model,
        labels_path=args.labels,
        conf=args.conf,
        imgsz=args.imgsz,
        stride=args.stride,
        H=H,
        W=W,
        zone_arg=args.zone,
    )

    logic = InteractionLogic()

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        foods, hands = perception.process_frame(frame)
        logic.process_frame(foods, hands)

        if args.show:
            for f in foods:
                x1, y1, x2, y2 = f["bbox"]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cx, cy = f["center"]
                cx, cy = int(cx), int(cy)

                color = (0, 255, 0) if f["inside"] else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 3, color, -1)
                label = f"ID {f['id']} {f['name']}"
                cv2.putText(
                    frame, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )

            for h in hands:
                hx1, hy1, hx2, hy2 = h["bbox"]
                hx1, hy1, hx2, hy2 = int(hx1), int(hy1), int(hx2), int(hy2)
                hcx, hcy = h["center"]
                hcx, hcy = int(hcx), int(hcy)

                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 255), 2)
                cv2.circle(frame, (hcx, hcy), 4, (255, 0, 255), -1)
                cv2.putText(
                    frame, "HAND", (hx1, max(0, hy1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1
                )

            perception.annotate_zone(frame)

            cv2.putText(
                frame,
                f"Counts: {logic.get_final_counts()}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Video Count", frame)
            cv2.moveWindow("Video Count", 800, 50)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_idx += 1

    cap.release()
    if args.show:
        cv2.destroyAllWindows()

    _print_final(logic.get_final_counts(), as_json=args.json)


if __name__ == "__main__":
    main()