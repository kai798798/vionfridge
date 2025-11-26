import argparse, json
import cv2
from ultralytics import YOLO
import supervision as sv
from perception import load_focus_labels, build_zone_and_annotator
from interaction import InteractionLogic
import mediapipe as mp

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
    p = argparse.ArgumentParser("count_video")  # Makefile command
    p.add_argument("video", help="Path to video file")
    p.add_argument("--show", action="store_true", help="Show live window with overlays")
    args = p.parse_args()

    args.labels = "data/labels.txt"
    args.model = "yolov8s.pt"
    args.conf = 0.25 # confidence threshold
    args.imgsz = 960 # image size
    args.stride = 1
    args.json = True # JSON output
    args.mode = "in"  # or net
    args.zone = "FULL" # full frame by default

    return args


def box_annotate_scene(frame, dets, labels, box_annot, label_annot):
    scene = box_annot.annotate(scene=frame, detections=dets)
    scene = label_annot.annotate(scene=scene, detections=dets, labels=labels)
    return scene


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
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # rewind

    if args.zone and args.zone.strip().upper() == "FULL":
        args.zone = f"0,0;{W},0;{W},{H};0,{H}"

    ZONE, annotate_zone = build_zone_and_annotator(H, W, args.zone)

    # YOLO
    model = YOLO(args.model)
    tracker = sv.ByteTrack()

    focus = load_focus_labels(args.labels)
    logic = InteractionLogic(mode=args.mode)

    box_annot = sv.BoxAnnotator()
    label_annot = sv.LabelAnnotator()

    # mediapipe
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )



    frame_idx = 0
    prev_det = None
    class_names = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        do_det = (frame_idx % args.stride == 0) or (prev_det is None)

        if do_det:
            res = model.predict(
                frame, imgsz=args.imgsz, conf=args.conf, verbose=False
            )[0]
            class_names = res.names
            dets = sv.Detections.from_ultralytics(res)
        else:
            dets = prev_det

        # focus filter
        if focus is not None and len(dets) > 0:
            keep = []
            for i in range(len(dets)):
                cid = None if dets.class_id is None else dets.class_id[i]
                name = (
                    class_names.get(int(cid), str(cid))
                    if cid is not None else "unknown"
                )
                if name in focus:
                    keep.append(i)
            dets = dets[keep]

        # track
        dets = tracker.update_with_detections(dets)
        prev_det = dets

        # inside/outside mask
        if len(dets):
            try:
                inside_mask = ZONE.trigger(dets)
            except Exception:
                centers = dets.get_anchors_coordinates(anchor=sv.Position.CENTER)
                inside_mask = [ZONE.contains(c) for c in centers]
        else:
            inside_mask = []

        # transitions -> counts (assume first seen = outside so appearing inside counts)
        for i in range(len(dets)):
            cid = None if dets.class_id is None else dets.class_id[i]
            name = (
                class_names.get(int(cid), str(cid))
                if cid is not None else "unknown"
            )
            tid = (
                None
                if getattr(dets, "tracker_id", None) is None
                else dets.tracker_id[i]
            )
            if tid is None:
                continue

            now_in = bool(inside_mask[i])
            logic.update(name, tid, now_in)

        if args.show:
            # show hands skeleton
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(img_rgb)
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        handLms,
                        mp_hands.HAND_CONNECTIONS,
                    )

            labels = []
            for i in range(len(dets)):
                conf = (
                    0.0
                    if getattr(dets, "confidence", None) is None
                    else float(dets.confidence[i])
                )
                cid = None if dets.class_id is None else dets.class_id[i]
                nm = (
                    class_names.get(int(cid), str(cid))
                    if cid is not None else "unknown"
                )
                tid = (
                    None
                    if getattr(dets, "tracker_id", None) is None
                    else dets.tracker_id[i]
                )
                lab = f"ID {tid if tid is not None else '-'} {nm} {conf:.2f}"
                labels.append(lab)

            frame_ = box_annotate_scene(frame, dets, labels, box_annot, label_annot)
            annotate_zone(frame_)
            cv2.putText(
                frame_,
                f"Counts: {logic.get_counts()}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Video Count (no-hands)", frame_)
            cv2.moveWindow("Video Count (no-hands)", 800, 50)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_idx += 1

    cap.release()
    if args.show:
        cv2.destroyAllWindows()

    _print_final(logic.get_counts(), as_json=args.json)


if __name__ == "__main__":
    main()