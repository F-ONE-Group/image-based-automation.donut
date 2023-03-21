import math

def validate_bbox(bb):
  """
  Ensures correct coordinates for bounding box. Returns true
  Returns
  -------
  True if bbox coordinates are valid. False otherwise.
  """
  try:
    if bb['xmin'] > bb['xmax']:
      return False
    if bb['ymin'] > bb['ymax']:
      return False
  except Exception as e:
    print(f"Invalid bbox: {bb}", e)
    return False
  return True

def get_center_distance(bb1, bb2):
    """
    Calculate the distance between the centers of two bounding boxes.
    Best case, distance between centers of predicted and ground truth bounding boxes will be 0.
    Worst case,  distance will be the larges diagonal in the screen - sqrt(1,1).

    Parameters
    ----------
    bb1 : dict
        Keys: {'xmin', 'xmax', 'ymin', 'ymax'}
        The (xmin, ymin) position is at the top left corner,
        the (xmax, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'xmin', 'xmax', 'ymin', 'ymax'}
        The (x, y) position is at the top left corner,
        the (xmax, ymax) position is at the bottom right corner

    Returns
    -------
    float
        in [0, sqrt(1+1)]
    """
    best_case = 0.0
    worst_case = math.sqrt(1+1) # max diagonal
    if not validate_bbox(bb1) or not validate_bbox(bb2):
      return worst_case

    # determine the coordinates of the center of each rectangle
    bb1_x_center = (bb1['xmax'] + bb1['xmin'])/2
    bb1_y_center = (bb1['ymax'] + bb1['ymin'])/2

    bb2_x_center = (bb2['xmax'] + bb2['xmin'])/2
    bb2_y_center = (bb2['ymax'] + bb2['ymin'])/2
    center_dist = math.sqrt((bb2_x_center - bb1_x_center)**2 + (bb2_y_center - bb1_y_center)**2)

    assert center_dist >= best_case
    assert center_dist <= worst_case # sqrt(1+1)
    return center_dist



def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Best case, IoU is 1 indicating perfect match between prediction and ground truth.
    Worst case, IoU is 0 when no overlap between bounding boxes.
    Modifed version from the following original on stackoverflow:
    https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation

    Parameters
    ----------
    bb1 : dict
        Keys: {'xmin', 'xmax', 'ymin', 'ymax'}
        The (xmin, ymin) position is at the top left corner,
        the (xmax, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'xmin', 'xmax', 'ymin', 'ymax'}
        The (x, y) position is at the top left corner,
        the (xmax, ymax) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    best_case = 1.0
    worst_case = 0.0
    if not validate_bbox(bb1) or not validate_bbox(bb2):
      return worst_case

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['xmin'], bb2['xmin'])
    y_top = max(bb1['ymin'], bb2['ymin'])
    x_right = min(bb1['xmax'], bb2['xmax'])
    y_bottom = min(bb1['ymax'], bb2['ymax'])

    # print(f"IoU x_left: {x_left}, y_top: {y_top}, x_right: {x_right}, y_bottom: {y_bottom}")

    if x_right < x_left or y_bottom < y_top:
        return worst_case # no bbox overlap

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # print(f"IoU intersection_area: {intersection_area}")

    # compute the area of both AABBs
    bb1_area = (bb1['xmax'] - bb1['xmin']) * (bb1['ymax'] - bb1['ymin'])
    bb2_area = (bb2['xmax'] - bb2['xmin']) * (bb2['ymax'] - bb2['ymin'])
    # print(f"IoU bb1_area: {bb1_area}")
    # print(f"IoU bb2_area: {bb2_area}")

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    # if iou > 0:
    #   print(f"IoU input bb1, bb2: {bb1} , {bb2}")
    #   print(f"IoU : {iou}")
    assert iou >= worst_case
    assert iou <= best_case
    return iou


def get_cui(bb1, bb2):
    """
    Calculates Central Distance times Union minus Intersection.
    The model should aim to minimize this function towards 0,
    which is complete bounding box overlap.

    Parameters
    ----------
    bb1 : dict
        Keys: {'xmin', 'xmax', 'ymin', 'ymax'}
        The (xmin, ymin) position is at the top left corner,
        the (xmax, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'xmin', 'xmax', 'ymin', 'ymax'}
        The (x, y) position is at the top left corner,
        the (xmax, ymax) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    best_case = 0.0
    # max(central distance) = sqrt(1+1)
    # max(U-I) = 1
    worst_case = 1*math.sqrt(1+1)
    # print(f"\n\n>>>>>>>> get_cui bb1: {bb1}, bb2 {bb2} \n\n")
    if not validate_bbox(bb1) or not validate_bbox(bb2):
      # print(f"get_cui: invalid boundig box: {bb1}, {bb2}")
      return worst_case

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['xmin'], bb2['xmin'])
    y_top = max(bb1['ymin'], bb2['ymin'])
    x_right = min(bb1['xmax'], bb2['xmax'])
    y_bottom = min(bb1['ymax'], bb2['ymax'])

    # print(f"get_cui: x_left: {x_left}, y_top: {y_top}, x_right: {x_right}, y_bottom: {y_bottom}")


    if x_right > x_left and y_bottom > y_top:
      # The intersection of two axis-aligned bounding boxes is always an
      # axis-aligned bounding box
      intersection_area = (x_right - x_left) * (y_bottom - y_top)
    else:
      intersection_area = 0.0
    # print(f"get_cui: intersection_area: {intersection_area}")


    # compute Union: the area of both AABBs
    bb1_area = (bb1['xmax'] - bb1['xmin']) * (bb1['ymax'] - bb1['ymin'])
    bb2_area = (bb2['xmax'] - bb2['xmin']) * (bb2['ymax'] - bb2['ymin'])
    union_area = float(bb1_area + bb2_area - intersection_area)
    # print(f"get_cui: bb1_area: {bb1_area}")
    # print(f"get_cui: bb2_area: {bb2_area}")
    # print(f"get_cui: union_area: {union_area}")

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    non_overlapping_area = float(union_area - intersection_area)
    # print(f"get_cui: non_overlapping_area: {non_overlapping_area}")
    center_distance = get_center_distance(bb1, bb2)
    # print(f"get_cui: center_distance: {center_distance}")
    cui = non_overlapping_area*center_distance
    # print(f"get_cui: CUI score: {cui}")
    assert cui >= best_case
    assert cui <= worst_case
    return cui