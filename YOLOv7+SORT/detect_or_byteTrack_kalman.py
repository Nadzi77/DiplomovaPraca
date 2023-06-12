import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from numpy.linalg import inv

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
                increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from bytetrack.byte_tracker import BYTETracker
from sort import *
from intersect import *
from filterpy.kalman import KalmanFilter

# python detect_or_byteTrack_kalman.py --weights yolov7.pt --no-trace --view-img --source inference\images\Jur_1_demo.mp4 --track --classes 0 1 2 3 5 6 7 16 --show-track --resultFPS 12
# python detect_or_byteTrack_kalman.py --weights yolov7.pt --no-trace --view-img --source inference\images\Jur_2_demo.mp4 --track --classes 0 --show-track          vidsLonger\Jur2_23_06_2022.mp4

"""Function to Draw Bounding boxes"""
def draw_boxes(img, bbox, identities=None, categories=None, confidences = None, names=None, colors = None):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        tl = opt.thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        # conf = confidences[i] if confidences is not None else 0

        color = colors[cat]
        
        if not opt.nobbox:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)

        if not opt.nolabel:
            label = str(id) + ":"+ names[cat] if identities is not None else  f'{names[cat]} {confidences[i]:.2f}'
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return img


def detect(save_img=False):
    obs_len = 12
    pred_len = 50
    warning_count = 0
    source, weights, view_img, save_txt, imgsz, trace, resultFPS = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.resultFPS
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    fps_label = int(resultFPS) if resultFPS != 2.5 else 2.5
    save_dir = save_dir.parent / f"{save_dir.name}_{fps_label}"
    if not opt.nosave:
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    ###################################
    trackTime = 0
    predictTime = 0
    color1 = (0,0,0)
    color2 = (250,0,0)
    color3 = (0, 0, 220)
    tracks_dict = {}
    # 2.5/6/12 FPS simulation
    xx = -1
    ###################################
    for path, img, im0s, vid_cap in dataset:
        xx += 1

        if resultFPS != 25:
            if resultFPS == 2.5:
                modFPSvariable = 10
            elif resultFPS == 6:
                modFPSvariable = 4
            else:
                # elif resultFPS == 12:
                modFPSvariable = 2

            if xx % modFPSvariable != 0:
                continue

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        pred_trajs_persons = []
        pred_trajs_cars = []

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                ###################################
                dets_to_sort = np.empty((0,6))
                # NOTE: We send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                np.array([x1, y1, x2, y2, conf, detclass])))


                if opt.track:
                    t0tr = time.time()
                    online_targets = bytetracker.update(dets_to_sort, im0.shape)
                    trackTime += time.time() - t0tr

                    online_targets = np.array(online_targets) 
                    # print(online_targets)
                    
                    # save first image
                    # if xx == 0:
                    #     cv2.imwrite('xxx1.png', im0.copy())

                    tids = []
                    for t in online_targets:
                        tlwh = [t[0], t[1], t[0] + (t[2] - t[0]) / 2, t[1] +  (t[3] - t[1]) / 2]
                        tid = str(int(t[4])) + '_' + str(int(t[5])) # id + class
                        tids.append(tid)
                        center_of_bbox = (int(tlwh[2]), int(tlwh[3]))

                        if tid in tracks_dict:
                            tracks_dict[tid].append(center_of_bbox)
                        else:
                            tracks_dict[tid] = [center_of_bbox]

                    for key, value in tracks_dict.copy().items():
                        if key not in tids:
                            del tracks_dict[key]
                        elif len(value) > obs_len + 1:
                            value.pop(0)

                    # draw boxes for visualization
                    if len(online_targets)>0:
                        bbox_xyxy = online_targets[:,:4]
                        identities = online_targets[:, 4]
                        categories = online_targets[:, 5]
                        confidences = None

                        
                        if opt.show_track:
                            #loop over tracks
                            for key, value in tracks_dict.items():
                                track_color = colors[0]

                                posi = value
                                [cv2.line(im0, (int(posi[i][0]),
                                                int(posi[i][1])), 
                                                (int(posi[i+1][0]),
                                                int(posi[i+1][1])),
                                                track_color, thickness=opt.thickness) 
                                                for i,_ in  enumerate(posi) 
                                                    if i < len(posi)-1 ]

                                t0pr = time.time()
                                if len(posi) >= obs_len:
                                    last8Posi = posi[-obs_len:]
                                    # kal = KalmanFilter(dim_x=4, dim_z=2)
                                    # kal.x = np.array([[0, 0, 0, 0]]).T  # x, y, vx, vy

                                    # kal.P = np.diag([1000, 1000, 1000, 1000])
                                    # kal.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
                                    # kal.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
                                    # kal.R = np.diag([1, 1])
                                    # kal.Q = np.diag([1, 1, 0.1, 0.1]) 


                                    # for i in range(8):
                                    #     position = last8Posi[i]
                                    #     kal.predict()
                                    #     kal.update([position[0], position[1]])

                                    # for i in range(12):
                                    #     old_pos = kal.x[:,0]
                                    #     kal.predict()
                                    #     new_pos = kal.x[:,0]
                                    #     cv2.line(im0, (int(old_pos[0]),
                                    #                    int(old_pos[1])), 
                                    #                   (int(new_pos[0]),
                                    #                    int(new_pos[1])),
                                    #                   color2, thickness=opt.thickness)

                                    #     # cv2.circle(im0, (int(kal.x[0,0]), int(kal.x[1,0])), 5, color2, 4)
                                    #     kal.update(kal.x[:2,:])

                                    sumX = np.sum(np.array(last8Posi), axis=0)[0]
                                    sumY = np.sum(np.array(last8Posi), axis=0)[1]
                                    prevX = sumX / obs_len
                                    prevY = sumY / obs_len
                                    actX = posi[-1][0]
                                    actY = posi[-1][1]
                                    pred_traj = [(actX, actY)]
                                    for i in range(pred_len):
                                        divider = obs_len / 2 - 0.5
                                        futX = actX + (i+1) * ((actX - prevX) / divider)
                                        futY = actY + (i+1) * ((actY - prevY) / divider)
                                        cv2.line(im0, (int(pred_traj[-1][0]),
                                                       int(pred_traj[-1][1])), 
                                                      (int(futX),
                                                       int(futY)),
                                                      color3, thickness=opt.thickness)
                                    pred_traj.append((futX, futY))
                                    if key.endswith("0"):
                                        pred_trajs_persons.append(pred_traj)
                                    else:
                                        pred_trajs_cars.append(pred_traj)
                                predictTime += time.time() - t0pr

                                        

                                
                else:
                    bbox_xyxy = dets_to_sort[:,:4]
                    identities = None
                    categories = dets_to_sort[:, 5]
                    confidences = dets_to_sort[:, 4]
                
                # im0 = draw_boxes(im0, bbox_xyxy, identities, categories, confidences, names, colors)
                ###################################


            for i in range(len(pred_trajs_persons)):
                for j in range(len(pred_trajs_cars)):
                    if doTrajectoriesIntersect(pred_trajs_persons[i], pred_trajs_cars[j]):
                        cv2.putText(im0, 'WARNING!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        warning_count += 1
            

            # Print time (inference + NMS)
            # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            # 2.5/6/12 FPS simulation
                            fps = resultFPS
                            # fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    print(f'Done. (Total time: {time.time() - t0:.3f}s, Tracking time: {trackTime:.2f}s, Predict time: {predictTime:.2f}s, Warnings count: {warning_count})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect_or_byteTrack_kalman', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    #######################################################
    parser.add_argument('--track', action='store_true', help='run tracking')
    parser.add_argument('--show-track', action='store_true', help='show tracked path')
    parser.add_argument('--show-fps', action='store_true', help='show fps')
    parser.add_argument('--thickness', type=int, default=2, help='bounding box and font size thickness')
    parser.add_argument('--seed', type=int, default=1, help='random seed to control bbox colors')
    parser.add_argument('--nobbox', action='store_true', help='don`t show bounding box')
    parser.add_argument('--nolabel', action='store_true', help='don`t show label')
    parser.add_argument('--unique-track-color', action='store_true', help='show each track in unique color')
    #######################################################
    parser.add_argument('--resultFPS', type=float, default=25, help='Result video FPS')
    #######################################################

    opt = parser.parse_args()
    print(opt)
    np.random.seed(opt.seed)

    bytetracker = BYTETracker(
        track_thresh=0.6,   # tracking confidence threshold
        match_thresh=0.8,   # matching threshold for tracking
        track_buffer=30,    # the frames for keep lost tracks
        frame_rate=30       # FPS
    )

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
