import numpy as np
import cv2
import time
from utilities import *
from moviepy.editor import VideoFileClip
from functools import reduce
import pickle
from scipy.ndimage.measurements import label
import sys


dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
print(dist_pickle)
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]


def get_rectangles(image, scales = [1, 1.5, 2, 2.5, 3], 
                   ystarts = [400, 400, 450, 450, 460], 
                   ystops = [528, 550, 620, 650, 700]):
    out_rectangles = []
    for scale, ystart, ystop in zip(scales, ystarts, ystops):
        rectangles = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        if len(rectangles) > 0:
            out_rectangles.append(rectangles)
    out_rectangles = [item for sublist in out_rectangles for item in sublist] 
    return out_rectangles


    

result_images = []
result_boxes = []
heatmap_images = []
result_img_all_boxes = []



class HeatHistory():
    def __init__(self):
        self.history = []
history = HeatHistory()
frames_to_remember=6
def pipeline(test_image):        
    rectangles = get_rectangles(test_image)
    all_boxes = draw_boxes(test_image, rectangles, color='random', thick=3)
    result_img_all_boxes.append(all_boxes)
    heatmap_image = np.zeros_like(test_image[:, :, 0])
    heatmap = add_heat(heatmap_image, rectangles)
    if len(history.history) >= frames_to_remember:
        history.history = history.history[1:]
    history.history.append(heatmap)
    heat_history = reduce(lambda h, acc: h + acc, history.history)/frames_to_remember
    ##threshold 2 was too high
    heatmap_image = apply_threshold(heat_history, 1)
    labels = label(heatmap_image)
    return draw_labeled_bboxes(np.copy(test_image), labels) , heatmap_image , all_boxes
        

def main():

    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    debug = sys.argv[3]

    
    
    output_1 = output_path +'_project_video1.mp4'
    output_2 = output_path + '_project_video2.mp4'
    output_3 = output_path + '_project_video3.mp4'
    clip1 = VideoFileClip(input_path)
    images = [i for i in clip1.iter_frames()]
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    size = images[0].shape[1], images[0].shape[0]
    writer_1 = cv2.VideoWriter(output_1, fourcc, 15, size)
    writer_2 = cv2.VideoWriter(output_2, fourcc, 15, size)
    writer_3 = cv2.VideoWriter(output_3, fourcc, 15, size)
    t=time.time()
    
    if debug == 'y' or debug == 'Y':
        for idx in range(0,len(images),2):
            final,HeatFinal,AllBoxesFinal = pipeline(images[idx])
            final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
            HeatFinal = np.uint8(HeatFinal)
            HeatFinal = cv2.cvtColor(HeatFinal, cv2.COLOR_GRAY2RGB)
            HeatFinal[:,:,0] = cv2.equalizeHist(HeatFinal[:,:,0])
            writer_3.write(final)
            writer_2.write(HeatFinal)
            writer_1.write(AllBoxesFinal)
            #cv2.imshow("Heat OUTPUT",HeatFinal)
            cv2.imshow("FINAL OUTPUT",final)
            # Display frame for X milliseconds and check if q key is pressed
            # q == quit
            temp=time.time()
            if(idx%100 == 0):
                print("Time taken: ",(temp-t)/60)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    
    
    else:
        for idx in range(0,len(images),2):
            final,HeatFinal,AllBoxesFinal = pipeline(images[idx])
            final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
            #HeatFinal = np.uint8(HeatFinal)
            #HeatFinal = cv2.cvtColor(HeatFinal, cv2.COLOR_GRAY2RGB)
            #HeatFinal[:,:,0] = cv2.equalizeHist(HeatFinal[:,:,0])
            writer_3.write(final)
            #writer_2.write(HeatFinal)
            #writer_1.write(AllBoxesFinal)
            #cv2.imshow("Heat OUTPUT",HeatFinal)
            cv2.imshow("FINAL OUTPUT",final)
            # Display frame for X milliseconds and check if q key is pressed
            # q == quit
            temp=time.time()
            if(idx%100 == 0):
                print("Time taken: ",(temp-t)/60)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    


if __name__ == '__main__':
    main()
