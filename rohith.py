import cv2
import numpy as np

def preprocess(frame):
    filtering = cv2.GaussianBlur(frame, (5,5),0)
    img_rotated= cv2.flip(filtering, 90)
    return img_rotated


def region_of_interest(processed_image, vertices):
    anding_img = np.zeros_like(processed_image)
    color_match = 255
    cv2.fillPoly(anding_img, vertices, color_match)
    stfu = cv2.bitwise_and(anding_img, processed_image)
    final_image_set = stfu
    return final_image_set

def lane_tracking(img, lines):
    copy_image= np.copy(img)
    merginig_img = np.zeros((img.shape[0],img.shape[1],3),
                          dtype = np.uint8)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(merginig_img,(x1,y1),(x2,y2),(10,255,255),
                     thickness=4)

    merged_img =cv2.addWeighted(copy_image,
                                0.8,
                                merginig_img,
                                1,
                                0.0)
    return  merged_img









