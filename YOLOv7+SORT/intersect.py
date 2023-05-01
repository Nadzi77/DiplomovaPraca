# Given three collinear points (x1, y1), (x2, y2), (x3, y3) the function checks if point (x2, y2) lies on line segment |(x1, y1),(x3, y3)| 
def onSegment(x1, y1, x2, y2, x3, y3):
    if ( (x2 <= max(x1, x3)) and (x2 >= min(x1, x3)) and (y2 <= max(y1, y3)) and (y2 >= min(y1, y3))):
        return True
    return False
  
def orientation(x1, y1, x2, y2, x3, y3):
    # to find the orientation of an ordered triplet ((x1, y1), (x2, y2), (x3, y3))
    # function returns the following values:
    # 0 : Collinear points
    # 1 : Clockwise
    # 2 : Counterclockwise
    val = (float(y2 - y1) * (x3 - x2)) - (float(x2 - x1) * (y3 - y2))
    if (val > 0):
        return 1 # Clockwise
    elif (val < 0):
        return 2 # Counterclockwise
    else:
        return 0 # Collinear
  
# returns true if the line segment '(x1, y1), (x2, y2)' and '(x3, y3), (x4, y4)' intersect.
def doSegmentIntersect(x1, y1, x2, y2, x3, y3, x4, y4):
    o1 = orientation(x1, y1, x2, y2, x3, y3)
    o2 = orientation(x1, y1, x2, y2, x4, y4)
    o3 = orientation(x3, y3, x4, y4, x1, y1)
    o4 = orientation(x3, y3, x4, y4, x2, y2)
  
    if ((o1 != o2) and (o3 != o4)):
        return True
  
    # Special Cases
    if ((o1 == 0) and onSegment(x1, y1, x3, y3, x2, y2)):
        return True
    if ((o2 == 0) and onSegment(x1, y1, x4, y4, x2, y2)):
        return True
    if ((o3 == 0) and onSegment(x3, y3, x1, y1, x4, y4)):
        return True
    if ((o4 == 0) and onSegment(x3, y3, x2, y2, x4, y4)):
        return True
  
    # If none of the cases
    return False

def doTrajectoriesIntersect(traj1, traj2):
    # trajs in shape [(x1,y1), (x2,y2), ....]
    for i in range(len(traj1) - 1):
        for j in range(len(traj2) - 1):
            if doSegmentIntersect(traj1[i][0], traj1[i][1], traj1[i+1][0], traj1[i+1][1], traj2[j][0], traj2[j][1], traj2[j+1][0], traj2[j+1][1]):
                return True
    return False