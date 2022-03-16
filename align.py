import numpy as np
import cv2

def get_bb_cor(txt):
  bb_cor = open(txt,'r')
  bb_cor = bb_cor.readlines()
  total_bb = []

  for line in bb_cor:
    cord = line.split(' ')
    obj_class, left, top, right, bottom = float(cord[0]), float(cord[1]), float(cord[2]), float(cord[3]), float(cord[4])
    bb_detected = [left, top, right, bottom]
    total_bb.append(bb_detected)

  return total_bb

def get_center_point(box):
    left, top, right, bottom = box  
    return left + ((right - left) // 2), top + ((bottom - top) // 2) # (x_c, y_c) # Need to fix bottom_left and bottom_right


def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect

def four_point_transform(image, pts):
	image = np.asarray(image)
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	return warped