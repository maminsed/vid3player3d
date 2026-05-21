import cv2
import numpy as np
import torch
from TennisProject.tracknet import BallTrackerNet
import torch.nn.functional as F
from TennisProject.postprocess import refine_kps
from TennisProject.homography import get_trans_matrix, refer_kps
from itertools import islice

class CourtDetectorNet():
    def __init__(self, path_model=None,  device='cuda'):
        self.model = BallTrackerNet(out_channels=15).to(device)
        self.device = device
        if path_model:
            self.model.load_state_dict(torch.load(path_model, map_location=device))
            self.model = self.model.to(device)
            self.model.eval()
    @torch.inference_mode()
    def infer_model(self, frames, start,end):
        scaleX = 1
        scaleY = 1
        
        kps_res = []
        matrixes_res = []
        for num_frame, image in enumerate(islice(frames, start,end), start): # tqdm
            inp_np = (image.astype(np.float32) / 255.)
            inp = torch.from_numpy(inp_np).permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)
            out = self.model(inp)[0]
            pred = F.sigmoid(out).cpu().numpy()

            points = []
            for kps_num in range(14):
                heatmap = (pred[kps_num]*255).astype(np.uint8)
                _, heatmap = cv2.threshold(heatmap, 170, 255, cv2.THRESH_BINARY)
                circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=2,
                                           minRadius=10, maxRadius=25)
                if circles is not None:
                    x_pred = circles[0][0][0]*scaleX
                    y_pred = circles[0][0][1]*scaleY
                    if kps_num not in [8, 12, 9]:
                        x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred), crop_size=40)
                    points.append((x_pred, y_pred))                
                else:
                    points.append(None)

            matrix_trans = get_trans_matrix(points) 
            points = None
            if matrix_trans is not None:
                points = cv2.perspectiveTransform(refer_kps, matrix_trans)
                matrix_trans = cv2.invert(matrix_trans)[1]
            kps_res.append(points)
            matrixes_res.append(matrix_trans)
            
        return matrixes_res, kps_res    
