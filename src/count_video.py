import argparse, json, time, collections
from pathlib import Path

import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

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
    p.add_argument("--labels", default="data/labels.txt",
                   help="Optional list of target labels (one per line). If present, only these are counted.")
    p.add_argument("--model", default="yolov8n.pt", help="Ultralytics model weights")
    p.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--stride", type=int, default=3, help="Detect every N frames (track in between)")
    p.add_argument("--show", action="store_true", help="Show live window with overlays")
    p.add_argument("--json", action="store_true", help="Print final counts as JSON")
    p.add_argument("--mode", choices=["in","net"], default="in",
                   help="'in': only +1 on enter; 'net': +1 enter, -1 exit")
    p.add_argument("--zone", type=str, default=None,
                   help="Custom zone polygon as x1,y1;x2,y2;...  or 'FULL' for whole frame")
    return p.parse_args()

def load_focus_labels(path: str):
    fp = Path(path)
    if not fp.exists():
        return None
    labels = [ln.strip() for ln in fp.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return set(labels) if labels else None

def parse_zone(zone_str: str):
    pts = []
    for pair in zone_str.split(";"):
        x, y = pair.split(",")
        pts.append((int(float(x)), int(float(y))))
    return pts

def box_annotate_scene(frame, dets, labels, box_annot, label_annot):
    scene = box_annot.annotate(scene=frame, detections=dets)
    scene = label_annot.annotate(scene=scene, detections=dets, labels=labels)
    return scene

def build_zone_and_annotator(H, W, zone_arg):
    # polygon
    if zone_arg:
        poly_np = np.array(parse_zone(zone_arg), dtype=int)
    else:
        y_top = int(H * 0.60)  # default: bottom 40%
        poly_np = np.array([[0, y_top], [W, y_top], [W, H], [0, H]], dtype=int)

    # PolygonZone (handle versions)
    try:
        zone = sv.PolygonZone(polygon=poly_np, frame_resolution_wh=(W, H))
    except TypeError:
        zone = sv.PolygonZone(polygon=poly_np)

    # Annotator (handle versions)
    def make_zone_annotator():
        try:
            color = getattr(sv.Color, "RED", None)
            ann = sv.PolygonZoneAnnotator(zone=zone, color=color) if color else sv.PolygonZoneAnnotator(zone=zone)
            def draw(frame):
                try:
                    ann.annotate(scene=frame)
                except TypeError:
                    ann.annotate(frame)
            return draw
        except TypeError:
            ann = sv.PolygonZoneAnnotator()
            def draw(frame):
                try:
                    ann.annotate(frame, zone)
                except TypeError:
                    ann.annotate(scene=frame, zone=zone)
            return draw
        except Exception:
            def draw(frame):
                pts = poly_np.reshape(-1, 1, 2)
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            return draw

    return zone, make_zone_annotator()

def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(
            f"Cannot open video: {args.video}\n"
            "Tip: if this is an iPhone HEVC .MOV or a corrupted MP4, re-mux or re-encode with ffmpeg."
        )

    ok, frame0 = cap.read()
    if not ok:
        raise SystemExit("Empty video")
    H, W = frame0.shape[:2]

    if args.zone and args.zone.strip().upper() == "FULL":
        args.zone = f"0,0;{W},0;{W},{H};0,{H}"

    ZONE, annotate_zone = build_zone_and_annotator(H, W, args.zone)

    model = YOLO(args.model)
    tracker = sv.ByteTrack()

    focus = load_focus_labels(args.labels)
    counts = collections.Counter()
    last_inside = {}

    box_annot = sv.BoxAnnotator()
    label_annot = sv.LabelAnnotator()

    frame_idx = 0
    prev_det = None
    class_names = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        do_det = (frame_idx % args.stride == 0) or (prev_det is None)

        if do_det:
            res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            class_names = res.names
            dets = sv.Detections.from_ultralytics(res)
        else:
            dets = prev_det

        # optional focus filter
        if focus is not None and len(dets) > 0:
            keep = []
            for i in range(len(dets)):
                cid = None if dets.class_id is None else dets.class_id[i]
                name = class_names.get(int(cid), str(cid)) if cid is not None else "unknown"
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
            name = class_names.get(int(cid), str(cid)) if cid is not None else "unknown"
            tid = None if getattr(dets, "tracker_id", None) is None else dets.tracker_id[i]
            if tid is None:
                continue

            now_in = bool(inside_mask[i])
            was_in = last_inside.get(tid, False)

            if now_in != was_in:
                if args.mode == "in":
                    if (not was_in and now_in): # outside -> inside
                        counts[name] += 1
                else:  # net
                    if (not was_in and now_in):
                        counts[name] += 1
                    else:
                        counts[name] -= 1
            last_inside[tid] = now_in

        if args.show:
            labels = []
            for i in range(len(dets)):
                conf = 0.0 if getattr(dets, "confidence", None) is None else float(dets.confidence[i])
                cid = None if dets.class_id is None else dets.class_id[i]
                nm = class_names.get(int(cid), str(cid)) if cid is not None else "unknown"
                tid = None if getattr(dets, "tracker_id", None) is None else dets.tracker_id[i]
                lab = f"ID {tid if tid is not None else '-'} {nm} {conf:.2f}"
                labels.append(lab)
            frame_ = box_annotate_scene(frame, dets, labels, box_annot, label_annot)
            annotate_zone(frame_)
            cv2.putText(frame_, f"Counts: {dict(counts)}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow("Video Count (no-hands)", frame_)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_idx += 1

    cap.release()
    if args.show:
        cv2.destroyAllWindows()

    _print_final(counts, as_json=args.json)

if __name__ == "__main__":
    main()