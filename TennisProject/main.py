import cv2
from vid3player.TennisProject.court_detection_net import CourtDetectorNet
import numpy as np
import torch
from vid3player.TennisProject.court_reference import CourtReference
from vid3player.TennisProject.bounce_detector import BounceDetector
from vid3player.TennisProject.person_detector import PersonDetector
from vid3player.TennisProject.ball_detector import BallDetector
from vid3player.TennisProject.utils import scene_detect
import argparse
import pandas as pd
import os.path as path

def read_video(path_video):
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    model_width,model_height = 640,360
    H,W = None,None
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if H is None: H,W = frame.shape[:2]
            frame = cv2.resize(frame, (model_width, model_height))
            frames.append(frame)
        else:
            break    
    cap.release()
    return frames, fps, W, H

def get_court_img():
    court_reference = CourtReference()
    court = court_reference.build_court_reference()
    court = cv2.dilate(court, np.ones((10, 10), dtype=np.uint8))
    court_img = (np.stack((court, court, court), axis=2)*255).astype(np.uint8)
    return court_img

def main(frames, num_scene, bounces, ball_track, homography_matrices, kps_court, persons_top, persons_bottom, data_path, start,end,
         draw_trace=False, trace=7, output_w=640, output_h=360,output_video_name:str=""):
    """
    :params
        frames: list of original images
        num_scene: index in scenes for the current start, end
        bounces: list of image numbers where ball touches the ground
        ball_track: list of (x,y) ball coordinates
        homography_matrices: list of homography matrices
        kps_court: list of 14 key points of tennis court
        persons_top: list of person bboxes located in the top of tennis court
        persons_bottom: list of person bboxes located in the bottom of tennis court
        start: start of the frame
        end: end of the frame
        draw_trace: whether to draw ball trace
        trace: the length of ball trace
    :return
        imgs_blocks: list of list of resulting images
    """ 
    imgs_res = []
    width_minimap = 166
    height_minimap = 350
    w,h = 640,360
    data = {
        'global_frame': [], # global frame number
        'local_frame': [], # global frame number
        'num_scene': [], # which number it belongs to
        'x_ball': [], # ball cordinates
        'y_ball': [], 
        'is_bounce': [], # whether it's a bounce
        'Iperson_top': [], # top person's bounding box: ((start_x, start_y), (end_x, end_y))
        'Iperson_bottom': [], # bottom person's bounding box: ((start_x, start_y), (end_x, end_y))
        'person_point_top': [], # top persons' point: (x,y)
        'person_point_bottom': [], # top persons' point: (x,y)
        'court_kps': [], # court's 14 key points
        'inv_matrix': [], # the inv matrix for the frame
        'output_video_name': [],
    }

    def extract_person_point(person,minimap,inv_mat):
        person_bbox = list(person[0])
        person_bbox = [(int(person_bbox[0] * output_w/w), int(person_bbox[1] * output_h/h)), (int(person_bbox[2] * output_w/w), int(person_bbox[3]*output_h/h))]

        # transmit person point to minimap
        person_point = list(person[1])
        person_point = np.array(person_point, dtype=np.float32).reshape(1, 1, 2)
        person_point = cv2.perspectiveTransform(person_point, inv_mat)
        person_point = (int(person_point[0, 0, 0]), int(person_point[0, 0, 1]))
        minimap = cv2.circle(minimap, person_point, radius=0, color=(255, 0, 0), thickness=80)
        return person_bbox,person_point,minimap
    # for num_scene in range(len(scenes)):
    #     sum_track = sum(is_track[scenes[num_scene][0]:scenes[num_scene][1]])
    #     len_track = scenes[num_scene][1] - scenes[num_scene][0]

    #     eps = 1e-15
    #     scene_rate = sum_track/(len_track+eps)
    #     if (scene_rate > 0.5):
    court_img = get_court_img()

    for i in range(start, end):
        img_res = cv2.resize(frames[i], (output_w,output_h))
        local_i = i - start
        inv_mat = homography_matrices[local_i]

        x_ball  = None
        y_ball = None
        is_bounce = False
        Iperson_top = None
        Iperson_bottom = None
        person_point_top = None
        person_point_bottom = None
        court_kps = None

        # draw ball trajectory
        if ball_track[local_i][0]:
            if draw_trace:
                for j in range(0, trace):
                    if local_i-j >= 0:
                        if ball_track[local_i-j][0]:
                            draw_x = int(ball_track[local_i-j][0]* output_w / w)
                            draw_y = int(ball_track[local_i-j][1]* output_h / h)
                            img_res = cv2.circle(frames[i], (draw_x, draw_y),
                            radius=3, color=(0, 255, 0), thickness=2)
            else:  
                x_ball = int(ball_track[local_i][0]* output_w / w)
                y_ball = int(ball_track[local_i][1]* output_h / h)
                img_res = cv2.circle(img_res , (x_ball, y_ball), radius=5,
                                        color=(0, 255, 0), thickness=2)
                img_res = cv2.putText(img_res, 'ball', 
                        org=(x_ball + 8, y_ball + 8),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8,
                        thickness=2,
                        color=(0, 255, 0))

        # draw court keypoints
        court_kps = []
        if kps_court[local_i] is not None:
            for j in range(len(kps_court[local_i])):
                point = (int(kps_court[local_i][j][0, 0] * output_w/w), int(kps_court[local_i][j][0, 1] * output_h/h))
                img_res = cv2.circle(img_res, point,
                                    radius=0, color=(0, 0, 255), thickness=10)
                court_kps.append(point)
        height, width, _ = img_res.shape

        # draw bounce in minimap
        if i in bounces and inv_mat is not None:
            is_bounce = True
            ball_point = ball_track[local_i]
            ball_point = np.array(ball_point, dtype=np.float32).reshape(1, 1, 2)
            ball_point = cv2.perspectiveTransform(ball_point, inv_mat)
            court_img = cv2.circle(court_img, (int(ball_point[0, 0, 0]), int(ball_point[0, 0, 1])),
                                                radius=0, color=(0, 255, 255), thickness=50)

        minimap = court_img.copy()

        # draw persons
        if len(persons_top[local_i]) > 0 and len(persons_top[local_i][0][0]) > 0:
            Iperson_top,person_point_top,minimap = extract_person_point(persons_top[local_i][0],minimap,inv_mat)
            img_res = cv2.rectangle(img_res, Iperson_top[0], Iperson_top[1], [255, 0, 0], 2)

        if len(persons_bottom[local_i]) > 0 and len(persons_bottom[local_i][0][0]) > 0:
            Iperson_bottom,person_point_bottom,minimap = extract_person_point(persons_bottom[local_i][0],minimap,inv_mat)
            img_res = cv2.rectangle(img_res, Iperson_bottom[0], Iperson_bottom[1], [255, 0, 0], 2) 
                
        # for j, person in enumerate(persons):
        #     if len(person[0]) > 0:
        #         person_bbox = list(person[0])
        #         person_bbox = [(int(person_bbox[0] * output_w/w), int(person_bbox[1] * output_h/h)), (int(person_bbox[2] * output_w/w), int(person_bbox[3]*output_h/h))]
        #         if Iperson_top is not None: Iperson_bottom = person_bbox
        #         else: Iperson_top = person_bbox
        #         img_res = cv2.rectangle(img_res, person_bbox[0], person_bbox[1], [255, 0, 0], 2)

        #         # transmit person point to minimap
        #         person_point = list(person[1])
        #         person_point = np.array(person_point, dtype=np.float32).reshape(1, 1, 2)
        #         person_point = cv2.perspectiveTransform(person_point, inv_mat)
        #         person_point = (int(person_point[0, 0, 0]), int(person_point[0, 0, 1]))
        #         person_points.append(person_point)
        #         minimap = cv2.circle(minimap, person_point, radius=0, color=(255, 0, 0), thickness=80)

        minimap = cv2.resize(minimap, (width_minimap, height_minimap))
        img_res[30:(30 + height_minimap), (width - 30 - width_minimap):(width - 30), :] = minimap
        imgs_res.append(img_res)
        
        data['global_frame'].append(i)
        data['local_frame'].append(local_i)
        data['x_ball'].append(x_ball)
        data['y_ball'].append(y_ball)
        data['is_bounce'].append(is_bounce)
        data['Iperson_top'].append(Iperson_top)
        data['Iperson_bottom'].append(Iperson_bottom)
        data['person_point_top'].append(person_point_top)
        data['person_point_bottom'].append(person_point_bottom)
        data['court_kps'].append(court_kps)
        data['inv_matrix'].append(inv_mat)
        data['num_scene'].append(num_scene)
        data['output_video_name'].append(output_video_name)

        # else: 
        #     if len_track < 10:
        #         imgs_res = imgs_res + frames[scenes[num_scene][0]:scenes[num_scene][1]] 
        #     else:
        #         imgs_block.append(imgs_res)
        #         imgs_res = []
        #         num_scene+=1
    if path.isfile(data_path):
        pd.DataFrame(data).to_csv(data_path, header=False,mode='a',index=False)
    else:
        pd.DataFrame(data).to_csv(data_path, header=True,mode='w',index=False)

    return imgs_res

def write(imgs_res, fps, path_output_video):
    height, width = imgs_res[0].shape[:2]
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    for num in range(len(imgs_res)):
        frame = imgs_res[num]
        out.write(frame)
    out.release()    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_ball_track_model', type=str, help='path to pretrained model for ball detection')
    parser.add_argument('--path_court_model', type=str, help='path to pretrained model for court detection')
    parser.add_argument('--path_bounce_model', type=str, help='path to pretrained model for bounce detection')
    parser.add_argument('--path_input_video', type=str, help='path to input video')
    parser.add_argument('--path_output_video', type=str, help='path to output video')
    parser.add_argument('--save', type=str, help='If we should save the video')
    args = parser.parse_args()
    if args.save == 'True':
        print("saving: ")
    else:
        print("not saving")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_video:str = args.path_input_video
    output_video:str = args.path_output_video
    print('scene detection')
    frames, fps, output_width,output_height = read_video(input_video) 
    scenes = scene_detect(input_video)    
    print(f"len(scenes): {len(scenes)}")

    print("initializing detectors")
    court_detector = CourtDetectorNet(args.path_court_model, device)
    ball_detector = BallDetector(args.path_ball_track_model, device)
    person_detector = PersonDetector(device)
    bounce_detector = BounceDetector(args.path_bounce_model)
    
    firstDash = output_video.find("-scene")
    dot = output_video.rfind(".")
    if firstDash == -1:
        base_name = output_video[:dot]
    else:
        base_name = output_video[:firstDash]
    data_path = base_name + ".csv"

    for num_scene in range(len(scenes)):
        output_video_name = f"{num_scene}-{output_video}"
        print(f"at num_scene: {num_scene}") 
        start,end = scenes[num_scene]  

        print('court detection')
        homography_matrices, kps_court = court_detector.infer_model(frames, start,end)

        # we don't want scenes that have less that half not court
        is_track = [x is not None for x in homography_matrices] 
        sum_track = sum(is_track)
        len_track = end - start
        eps = 1e-15
        scene_rate = sum_track/(len_track+eps)
        if (scene_rate <= 0.5): continue

        print('ball detection')
        ball_track = ball_detector.infer_model(frames, start, end)

        print('person detection')
        persons_top, persons_bottom = person_detector.track_players(frames, homography_matrices, start, end, filter_players=True)

        # bounce detection
        print('bounce detection')
        x_ball = [x[0] for x in ball_track]
        y_ball = [x[1] for x in ball_track]
        bounces = bounce_detector.predict(x_ball, y_ball, start=start)

        print('combining')
        imgs_res = main(frames, num_scene, bounces, ball_track, homography_matrices, kps_court, persons_top, persons_bottom, data_path, start,end,
                        draw_trace=False, output_w=output_width, output_h=output_height,output_video_name=output_video_name)
        
        del homography_matrices, ball_track, bounces, persons_bottom, persons_top, kps_court
        if args.save == 'True':
            print('downloading')
            write(imgs_res, fps, output_video_name)
 

    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Peak memory usage: {round(usage / 1_000_000,2)} GB")



