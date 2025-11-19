from pathlib import Path
import numpy as np
import cv2
import supervision as sv

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
    # polygon
    if zone_arg:
        poly_np = np.array(parse_zone(zone_arg), dtype=int)
    else:
        y_top = int(H * 0.60)  # default: bottom 40%
        poly_np = np.array([[0, y_top], [W, y_top], [W, H], [0, H]], dtype=int)

    try: # PolygonZone (handle versions)
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