def get_center(bbox):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (center_x, center_y)
def get_foot(bbox):
    x1, y1, x2, y2 = bbox
    foot_x = (x1 + x2) / 2
    foot_y = y2
    return (foot_x, foot_y)