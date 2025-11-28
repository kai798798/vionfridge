from interaction import InteractionLogic

def make_food(tid, name, inside, bbox):
    """Helper to create a fake food item."""
    # center is calculated roughly from bbox for the logic
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return {
        "id": tid,
        "name": name,
        "inside": inside,
        "bbox": bbox,
        "center": (cx, cy)
    }

def make_hand(bbox):
    """Helper to create a fake hand."""
    return {
        "bbox": bbox,
        "center": (0,0) # Center doesn't matter for overlap check
    }

def run_test():
    print("=== TESTING INTERACTION LOGIC ===")
    logic = InteractionLogic()

    # ---------------------------------------------------------
    # SCENARIO 1: The Valid Banana Entry
    # A banana moves from OUT -> IN while being held by a hand.
    # ---------------------------------------------------------
    print("\n[Scenario 1] Banana moves OUT -> IN with Hand contact...")
    
    # Frame 1: Banana is OUT, Hand is touching it
    # Hand Box: (100, 100) to (200, 200)
    # Food Box: (120, 120) to (150, 150) -> INSIDE HAND BOX
    hand = make_hand((100, 100, 200, 200)) 
    banana_f1 = make_food(1, "banana", inside=False, bbox=(120, 120, 150, 150))
    
    logic.process_frame([banana_f1], [hand])
    
    # Frame 2: Banana crosses boundary to IN, Hand still touching
    banana_f2 = make_food(1, "banana", inside=True, bbox=(130, 130, 160, 160))
    
    logic.process_frame([banana_f2], [hand])
    
    # CHECK RESULTS
    counts = logic.get_final_counts()
    print(f"Result: {counts}")
    if counts["banana"]["in"] == 1:
        print("PASS: Banana counted IN correctly.")
    else:
        print("FAIL: Banana was not counted.")

    # ---------------------------------------------------------
    # SCENARIO 2: The Ghost Apple (No Hand)
    # An apple moves OUT -> IN, but NO hand is detected.
    # This simulates a glitch or the item rolling on its own.
    # ---------------------------------------------------------
    print("\n[Scenario 2] Apple moves OUT -> IN *without* Hand...")

    # Frame 1: Apple OUT
    apple_f1 = make_food(2, "apple", inside=False, bbox=(300, 300, 350, 350))
    logic.process_frame([apple_f1], []) # Empty hands list

    # Frame 2: Apple IN (Crossed zone)
    apple_f2 = make_food(2, "apple", inside=True, bbox=(310, 310, 360, 360))
    logic.process_frame([apple_f2], []) # Empty hands list

    # CHECK RESULTS
    counts = logic.get_final_counts()
    print(f"Result: {counts}")
    if counts["apple"]["in"] == 0:
        print("PASS: Ghost Apple was IGNORED (Correct).")
    else:
        print("FAIL: Ghost Apple was wrongly counted.")

    # ---------------------------------------------------------
    # SCENARIO 3: The "Taking Out" Sandwich
    # A sandwich moves IN -> OUT with Hand contact.
    # ---------------------------------------------------------
    print("\n[Scenario 3] Sandwich moves IN -> OUT with Hand...")

    # Frame 1: Sandwich IN
    hand = make_hand((100, 100, 200, 200))
    sand_f1 = make_food(3, "sandwich", inside=True, bbox=(120, 120, 150, 150))
    logic.process_frame([sand_f1], [hand])

    # Frame 2: Sandwich OUT
    sand_f2 = make_food(3, "sandwich", inside=False, bbox=(120, 120, 150, 150))
    logic.process_frame([sand_f2], [hand])

    # CHECK RESULTS
    counts = logic.get_final_counts()
    print(f"Result: {counts}")
    if counts["sandwich"]["out"] == 1:
        print("PASS: Sandwich counted OUT correctly.")
    else:
        print("FAIL: Sandwich was not counted.")

if __name__ == "__main__":
    run_test()