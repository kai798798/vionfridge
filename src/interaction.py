import collections

class InteractionLogic:
    def __init__(self, mode="in"):
        self.mode = mode  # "in", "net", "cross"
        self.last_inside = {}  # tracker_id -> bool
        self.counts = collections.Counter()

    def update(self, name, tracker_id, now_in):
        """
        Call this once per detection per frame.

        name: class name, e.g. "banana"
        tracker_id: stable ID from ByteTrack
        now_in: bool, whether it's inside the zone this frame
        """
        was_in = self.last_inside.get(tracker_id, False)

        if now_in != was_in:
            if self.mode == "in":
                if not was_in and now_in:  # outside -> inside
                    self.counts[name] += 1
            elif self.mode == "net":
                if not was_in and now_in:
                    self.counts[name] += 1
                else:
                    self.counts[name] -= 1
            elif self.mode == "cross":
                # every boundary crossing
                self.counts[name] += 1

        self.last_inside[tracker_id] = now_in

    def get_counts(self):
        return dict(self.counts)