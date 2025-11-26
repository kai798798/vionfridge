from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
from pathlib import Path
import mediapipe as mp


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


def build_zone_and_annotator(H, W, zone_arg):
    if zone_arg:
        poly_np = np.array(parse_zone(zone_arg), dtype=int)
    else:
        # default: full frame
        poly_np = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=int)

    try: # versions
        zone = sv.PolygonZone(polygon=poly_np, frame_resolution_wh=(W, H))
    except TypeError:
        zone = sv.PolygonZone(polygon=poly_np)

    def make_zone_annotator():
        """
        Annotator (handle versions)
        """
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


class Perception:
    def __init__(self, model_path, labels_path, conf, imgsz, stride, H, W, zone_arg):
        # yolo tracking
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.focus = load_focus_labels(labels_path)
        self.conf = conf
        self.imgsz = imgsz
        self.stride = stride
        self.frame_idx = 0
        self.prev_det = None
        self.class_names = None

        # zone setup
        if zone_arg and zone_arg.strip().upper() == "FULL":
            zone_arg = f"0,0;{W},0;{W},{H};0,{H}"
        self.ZONE, self.annotate_zone = build_zone_and_annotator(H, W, zone_arg)

        # mediapipe hand
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _detect_foods(self, frame):
        """
        YOLO + ByteTrack + zone -> list of food dicts.
        """
        do_det = (self.frame_idx % self.stride == 0) or (self.prev_det is None)

        if do_det:
            res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
            self.class_names = res.names
            dets = sv.Detections.from_ultralytics(res)
        else:
            dets = self.prev_det

        # focus filter
        if self.focus is not None and len(dets) > 0:
            keep = []
            for i in range(len(dets)):
                cid = None if dets.class_id is None else dets.class_id[i]
                name = (
                    self.class_names.get(int(cid), str(cid))
                    if cid is not None else "unknown"
                )
                if name in self.focus:
                    keep.append(i)
            dets = dets[keep]

        dets = self.tracker.update_with_detections(dets)
        self.prev_det = dets

        if len(dets):
            try:
                inside_mask = self.ZONE.trigger(dets)
            except Exception:
                centers = dets.get_anchors_coordinates(anchor=sv.Position.CENTER)
                inside_mask = [self.ZONE.contains(c) for c in centers]
        else:
            inside_mask = []

        foods = []
        for i in range(len(dets)):
            cid = None if dets.class_id is None else dets.class_id[i]
            name = (
                self.class_names.get(int(cid), str(cid))
                if cid is not None else "unknown"
            )
            tid = (
                None
                if getattr(dets, "tracker_id", None) is None
                else dets.tracker_id[i]
            )
            if tid is None:
                continue

            x1, y1, x2, y2 = dets.xyxy[i]
            cx, cy = ((x1 + x2) / 2, (y1 + y2) / 2)
            foods.append(
                {
                    "id": int(tid),
                    "name": name,
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "center": (float(cx), float(cy)),
                    "inside": bool(inside_mask[i]),
                }
            )

        return foods

    def _detect_hands(self, frame):
        """
        MediaPipe Hands -> list of hand dicts:
          { "bbox": (x1,y1,x2,y2), "center": (cx,cy) }
        """
        H, W = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        hands = []
        if not results.multi_hand_landmarks:
            return hands

        for handLms in results.multi_hand_landmarks:
            xs = [lm.x * W for lm in handLms.landmark]
            ys = [lm.y * H for lm in handLms.landmark]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            hands.append(
                {
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "center": (float(cx), float(cy)),
                }
            )

        return hands

    def process_frame(self, frame):
        """
        Returns:
            foods: list of dicts:
              {
                "id": tid,
                "name": "banana",
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "inside": bool
              }
            hands: list of dicts:
              {
                "bbox": (hx1, hy1, hx2, hy2),
                "center": (hx, hy)
              }
        """
        foods = self._detect_foods(frame)
        hands = self._detect_hands(frame)

        self.frame_idx += 1
        return foods, hands