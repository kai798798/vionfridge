import collections
from typing import List, Dict, Any


class InteractionLogic:
    def __init__(
        self,
        max_missing_frames: int = 30,
        move_threshold_pixels: float = 15.0,
    ):
        self.item_states: Dict[int, Dict[str, Any]] = {}
        self.final_counts: Dict[str, Dict[str, int]] = collections.defaultdict(
            lambda: {"in": 0, "out": 0}
        )

        self.max_missing_frames = max_missing_frames
        self.move_threshold_pixels = move_threshold_pixels
        self.frame_count = 0
        self._finalized = False

    def process_frame(self, foods: List[Dict[str, Any]], hands: List[Dict[str, Any]]):
        self.frame_count += 1
        seen_ids = set()

        for food in foods:
            tracker_id = food["id"]
            name = food["name"]
            cx, cy = food["center"]

            seen_ids.add(tracker_id)
            state = self.item_states.get(tracker_id)

            if state is None:
                # first time we see this object
                self.item_states[tracker_id] = {
                    "name": name,
                    "first_center": (cx, cy),
                    "last_center": (cx, cy),
                    "last_seen": self.frame_count,
                }
                if name not in self.final_counts:
                    self.final_counts[name] = {"in": 0, "out": 0}
            else:
                state["last_center"] = (cx, cy)
                state["last_seen"] = self.frame_count

        ids_to_remove = []
        for tid, state in self.item_states.items():
            frames_since_seen = self.frame_count - state["last_seen"]
            if frames_since_seen > self.max_missing_frames:
                self._finalize_one(tid, state)
                ids_to_remove.append(tid)

        for tid in ids_to_remove:
            del self.item_states[tid]

    def _finalize_one(self, tid: int, state: Dict[str, Any]):
        name = state["name"]
        (fx, fy) = state["first_center"]
        (lx, ly) = state["last_center"]
        dx = lx - fx

        if dx > self.move_threshold_pixels:
            self.final_counts[name]["in"] += 1
            print(f"[{tid} / {name}] FINALIZED: IN (dx={dx:.1f})")
        elif dx < -self.move_threshold_pixels:
            self.final_counts[name]["out"] += 1
            print(f"[{tid} / {name}] FINALIZED: OUT (dx={dx:.1f})")

    def _finalize_remaining(self):
        if self._finalized:
            return
        for tid, state in list(self.item_states.items()):
            self._finalize_one(tid, state)
        self.item_states.clear()
        self._finalized = True

    def peek_counts(self):
        return dict(self.final_counts)

    def get_final_counts(self):
        self._finalize_remaining()
        return dict(self.final_counts)








"""
import collections
from typing import List, Dict, Tuple, Any

class InteractionLogic:
    def __init__(self, max_missing_frames=30):
        
        max_missing_frames: How many frames an ID can be missing before we delete it 
                            from memory (to save RAM).
        
        self.item_states: Dict[int, Dict[str, Any]] = {}
        self.final_counts: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: {"in": 0, "out": 0})
        
        # Cleanup settings
        self.max_missing_frames = max_missing_frames
        self.frame_count = 0 

    def _is_overlapping(self, bbox_a, bbox_b):
        x1_a, y1_a, x2_a, y2_a = bbox_a
        x1_b, y1_b, x2_b, y2_b = bbox_b
        if x1_a > x2_b or x2_a < x1_b or y1_a > y2_b or y2_a < y1_b:
            return False
        return True

    def _is_being_handled(self, food_bbox, hands):
        for hand in hands:
            if self._is_overlapping(food_bbox, hand["bbox"]):
                return True
        return False

    def process_frame(self, foods: List[Dict[str, Any]], hands: List[Dict[str, Any]]):
        self.frame_count += 1
        
        # 1. Update states and check for counting
        for food in foods:
            tracker_id = food["id"]
            name = food["name"]
            now_in = food["inside"]
            
            state = self.item_states.get(tracker_id)
            if state is None:
                self.item_states[tracker_id] = {
                    "name": name,
                    "was_in": now_in,
                    "last_center": food["center"],
                    "last_seen": self.frame_count  # NEW: Track when we last saw it
                }
                if name not in self.final_counts:
                    self.final_counts[name] = {"in": 0, "out": 0}
                continue

            # Update last seen time
            state["last_seen"] = self.frame_count # NEW

            was_in = state["was_in"]
            is_handled = self._is_being_handled(food["bbox"], hands)

            if now_in != was_in:
                if is_handled:
                    if not was_in and now_in:
                        self.final_counts[name]["in"] += 1
                        print(f"[{tracker_id} / {name}] COUNTED: IN (Handled)")
                    elif was_in and not now_in:
                        self.final_counts[name]["out"] += 1
                        print(f"[{tracker_id} / {name}] COUNTED: OUT (Handled)")
                
                state["was_in"] = now_in
            
            state["last_center"] = food["center"]

        # 2. CLEANUP LOGIC (Garbage Collection)
        ids_to_remove = []
        
        for tid, state in self.item_states.items():
            # Calculate how long it has been gone
            frames_since_seen = self.frame_count - state["last_seen"]
            
            if frames_since_seen > self.max_missing_frames:
                ids_to_remove.append(tid)

        for tid in ids_to_remove:
            # Optional: Print when deleting so you know it happened
            # print(f"Removing ID {tid} (inactive for {self.max_missing_frames} frames)")
            del self.item_states[tid]
            
    def get_final_counts(self):
        return dict(self.final_counts)
"""