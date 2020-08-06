
from model import detect, filter_boxes, detr, transform
from model import CLASSES, DEVICE
from PIL import Image
import time
import numpy as np
import cv2
from PIL import Image
from matplotlib import cm


def main():
    cap = cv2.VideoCapture('cabc30fc-e7726578.mov')

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if ret==True:
            # frame = cv2.flip(frame,0)

            # write the flipped frame
            out.write(frame)
            # im = Image.fromarray(np.uint8(cm.gist_earth(frame)))
            scores,boxes=run_model(im=frame,iou=0.85,confidence=0.96)
            for i in range(boxes.shape[0]):
                class_id = scores[i].argmax()
                label = CLASSES[class_id]
                confidence = scores[i].max()
                x0, y0, x1, y1 = boxes[i]
                frame = cv2.rectangle(frame, (x0, y0), (x1, y1), (0,0,255), 3) 

            cv2.imshow('frame',frame)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def run_model(im,iou, confidence):
    apply_nms = 'enabled'
    tstart = time.time()
    
    scores, boxes = detect(im, detr, transform, device=DEVICE)
    scores, boxes = filter_boxes(scores, boxes, confidence=confidence, iou=iou, apply_nms=apply_nms)
    
    scores = scores.data.numpy()
    boxes = boxes.data.numpy()

    tend = time.time()

    existing_classes = set()
    return scores,boxes


if __name__=="__main__":
    main()