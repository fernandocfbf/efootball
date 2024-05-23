def define_center_point(box):
    x1, y1 = box[0], box[1]
    x2, y2 = box[2], box[3]
    x_center = (x1+x2)/2
    y_center = (y1+y2)/2
    return [int(x_center), int(y_center)]