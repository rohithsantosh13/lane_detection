import cv2
import rohith
import numpy as np
videoPath = 'project_video.mp4'
cap = cv2.VideoCapture(videoPath)

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

while(True):


     ret,frame = cap.read()

     processed_image = rohith.preprocess(frame)

     width = processed_image.shape[0]
     height = processed_image.shape[1]

     region_of_interest_vertices = [
          (0, width), (height / 2, width / 2), (height, width)
     ]

     gray_scale = cv2.cvtColor(processed_image,cv2.COLOR_BGR2GRAY)
     edge_detection = cv2.Canny(gray_scale, 55, 150)
     cropped_image = rohith.region_of_interest(edge_detection,np.array([region_of_interest_vertices],np.int32))



     lane_tracking1 = cv2.HoughLinesP(cropped_image,
                                     rho=6, theta=np.pi / 60,
                                     threshold=160,
                                     lines=np.array([]),
                                     minLineLength=40,
                                     maxLineGap=25)

     line_added_img = lane_tracking(processed_image,lane_tracking1)



     cv2.imshow('edge detected image with only lanes',cropped_image)
     cv2.imshow('video_with_lane_lines', line_added_img)



     if cv2.waitKey(1) & 0xFF == ord('p'):
         break
cap.release()
cv2.destroyAllWindows()

