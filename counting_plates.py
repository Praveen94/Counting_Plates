# importing modules

import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils

import math
# capturing video through webcam
cap = cv2.VideoCapture(0)
count = 0
old_value = 0
new_value = 0
found = False

plate_num=0
while True:
    _, image = cap.read()
    pixel=image.copy()
 
    save_img = image.copy()

    fixed_point = (int(185), int(290))

    cv2.circle(image, (fixed_point[0], fixed_point[1]), 3, (0, 0, 0), -3)

    shift1 = pixel[fixed_point[1],fixed_point[0]]

    (r,g,b) = shift1

    shift1 = hex(r) + hex(g)[2:] + hex(b)[2:]

    decimal = int(shift1, 16)

    image = image[200:478, 0:639]

    new_value = decimal

    difference = old_value - new_value

    if difference > 2500000:

        plate_num = plate_num + 1
        print("this is plate", plate_num)
        cv2.circle(save_img, (fixed_point[0], fixed_point[1]), 3, (0, 0, 0), -3)
        found = True

    else:
        found = False

    old_value = decimal
    height, width = pixel.shape[:2]
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 15, 75, 75)
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None

    # loop over the contours individually
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 1000:
            continue

        # compute the rotated bounding box of the contour
        orig = image.copy()
        
        sorted_contours_x = sorted(c, key=lambda k: [k[0][0], k[0][1]])
        sorted_contours_y = sorted(c, key=lambda k: [k[0][1], k[0][0]])

        cv2.circle(orig, (int(sorted_contours_y[0][0][0]), int(sorted_contours_y[0][0][1])), 5, (0, 0, 255),
                   -1)  # top left
        cv2.circle(orig, (int(sorted_contours_x[-1][0][0]), int(sorted_contours_x[-1][0][1])), 5, (0, 0, 255),
                   -1)  # top right
        cv2.circle(orig, (int(sorted_contours_y[-1][0][0]), int(sorted_contours_y[-1][0][1])), 5, (0, 0, 255),
                   -1)  # bottom right
        cv2.circle(orig, (int(sorted_contours_x[0][0][0]), int(sorted_contours_x[0][0][1])), 5, (0, 0, 255),
                   -1)  # bottom left

        top_left = (int(sorted_contours_y[0][0][0]), int(sorted_contours_y[0][0][1]))

        tl=(int(sorted_contours_y[0][0][0]), int(sorted_contours_y[0][0][1]))


        bl=(int(sorted_contours_x[0][0][0]),int(sorted_contours_x[0][0][1]))


        tr=(int(sorted_contours_x[-1][0][0]),int(sorted_contours_x[-1][0][1]))

        br=(int(sorted_contours_y[-1][0][0]),int(sorted_contours_y[-1][0][1]))

        cv2.line(orig, (int(tl[0]), int(tl[1])), (int(br[0]), int(br[1])),
                 (255, 0, 255), 2)

        cv2.line(orig, (int(tr[0]), int(tr[1])), (int(bl[0]), int(bl[1])),
                 (255, 0, 255), 2)


        cv2.line(edged, (int(tl[0]), int(tl[1])), (int(br[0]), int(br[1])),
                 (255, 0, 255), 2)

        cv2.line(edged, (int(tr[0]), int(tr[1])), (int(bl[0]), int(bl[1])),
                 (255, 0, 255), 2)



        cv2.putText(edged, "Plate{}".format(plate_num),
                    (int(top_left[0] - 15), int(top_left[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)

        cv2.imshow("Edged", edged)

       
        cv2.drawContours(orig, c, -1, (0, 255, 0), 3)
        if found:

            path = 'Counting_Plates/plates/plate' + str(plate_num) + '.png'
            cv2.putText(orig, "Plate{}".format(plate_num),
                        (int(top_left[0] - 15), int(top_left[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)
            cv2.imwrite(path,orig)

        cv2.circle(image, (fixed_point[0], fixed_point[1]), 2, (0, 0, 0), -3)

        cv2.imshow("Image", orig)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

print("Total Plates",plate_num)
