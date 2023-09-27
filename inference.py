import numpy as np
import cv2
import torch
from glob import glob
import os
from model import create_model
import pandas as pd
from config import TRAIN_TEST_VALID_DIR, OUT_DIR, model_path, CLASSES, detection_threshold, iou_thr, map_name
from xml.etree import ElementTree as et
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def set_up_computation_device():
    try:
        import torch
    except ModuleNotFoundError:
        pass

    # set the computation device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # load the model and the trained weights
    model = create_model(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device
    ))

    model.eval()

    return model

# code from
# https://github.com/vineeth2309/IOU/tree/main and
# https://medium.com/analytics-vidhya/non-max-suppression-nms-6623e6572536
def IOU(coords1, coords2):
    """ min coords are top left hand corner:
        coords1 = (xmin, ymin, xmax, ymax), and coords2 = (xmin2, ymin2, xmax2, ymax2) """

    x1, y1, x2, y2 = coords1
    
    x3, y3, x4, y4 = coords2

    x_inter1 = max(x1, x3)
    y_inter1 = max(y1, y3)
    x_inter2 = min(x2, x4)
    y_inter2 = min(y2, y4)
    
    width_inter = x_inter2 - x_inter1
    height_inter = y_inter2 - y_inter1
    
    # rejecting non overlapping boxes
    if height_inter <= 0 or width_inter <= 0:
        return
    
    area_inter = width_inter * height_inter
    
    area1 = (x2-x1) * (y2-y1)
    area2 = (x4-x3) * (y4-y3)
    area_union = (area1 + area2) - abs(area_inter)
    
    iou = area_inter / area_union
    return iou


def get_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): (formatted like `gt_boxes`)
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """

    all_pred_indices= range(len(pred_boxes))
    all_gt_indices=range(len(gt_boxes))
    if len(all_pred_indices)==0:
        tp=0
        fp=0
        fn=len(gt_boxes)
        return tp, fp, fn
    if len(all_gt_indices)==0:
        tp=0
        fp=len(pred_boxes)
        fn=0
        return tp, fp, fn
 
    gt_idx_thr=[]
    pred_idx_thr=[]
    ious=[]
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
         
            # only calculate iou if same class
            if pred_box[1] == gt_box[1]:
                iou= IOU(gt_box[0], pred_box[0])
                
                if iou == None:
                    iou = 0
    
                if iou >iou_thr:
                    gt_idx_thr.append(igb)
                    pred_idx_thr.append(ipb)
                    ious.append(iou)

    iou_sort = np.argsort(ious)[::1]

    if len(iou_sort)==0:
        tp=0
        fp=len(pred_boxes)
        fn=len(gt_boxes)
        return tp, fp, fn
    else:
        gt_match_idx=[]
        pred_match_idx=[]
        for idx in iou_sort:
            gt_idx=gt_idx_thr[idx]
            pr_idx= pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if(gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp= len(gt_match_idx)
        fp= len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)
    return tp, fp, fn


def get_gt_boxes(DIR_TEST):
    boxes = []

    images = glob(os.path.join(DIR_TEST, '*.png'))

    for i in images:

        image_name = i.split('\\')[-1].split('.')[0]
        annot_filename = image_name + '.xml'
        annot_file_path = os.path.join(DIR_TEST, annot_filename)

        
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        xml_width = int(root.find('size').find('width').text)
        xml_height= int(root.find('size').find('height').text)

        image_width = 512
        image_height = 512

        for member in root.findall('object'):
            # xmin = left corner x-coordinates
            xmin = int(member.find('bndbox').find('xmin').text)
            # xmax = right corner x-coordinates
            xmax = int(member.find('bndbox').find('xmax').text)
            # ymin = left corner y-coordinates
            ymin = int(member.find('bndbox').find('ymin').text)
            # ymax = right corner y-coordinates
            ymax = int(member.find('bndbox').find('ymax').text)
                    
            # resize the bounding boxes according to the...
            # ... desired `width`, `height`
            xmin_final = (xmin/xml_width)*image_width
            xmax_final = (xmax/xml_width)*image_width
            ymin_final = (ymin/xml_height)*image_height
            ymax_final = (ymax/xml_height)*image_height

            cl = member.find('name').text
                    
            boxes.append([[xmin_final, ymin_final, xmax_final, ymax_final], cl])

    return boxes


def get_gt_boxes_single(DIR_TEST, image_name):
    boxes = []
    cls = []

    annot_filename = image_name + '.xml'
    annot_file_path = os.path.join(DIR_TEST, annot_filename)

        
    tree = et.parse(annot_file_path)
    root = tree.getroot()

    xml_width = int(root.find('size').find('width').text)
    xml_height= int(root.find('size').find('height').text)

    image_width = 512
    image_height = 512

    for member in root.findall('object'):
        # xmin = left corner x-coordinates
        xmin = int(member.find('bndbox').find('xmin').text)
        # xmax = right corner x-coordinates
        xmax = int(member.find('bndbox').find('xmax').text)
        # ymin = left corner y-coordinates
        ymin = int(member.find('bndbox').find('ymin').text)
        # ymax = right corner y-coordinates
        ymax = int(member.find('bndbox').find('ymax').text)
                    
        # resize the bounding boxes according to the...
        # ... desired `width`, `height`
        xmin_final = (xmin/xml_width)*image_width
        xmax_final = (xmax/xml_width)*image_width
        ymin_final = (ymin/xml_height)*image_height
        ymax_final = (ymax/xml_height)*image_height

        cl = member.find('name').text
        cl = CLASSES.index(cl)
        cls.append(cl)
                    
        boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

    return boxes, cls


def adjust_pixel_coords(bbox_x, bbox_y, image_name, path):
    df = pd.read_csv(os.path.join(path, 'img_coords.csv'))

    corner_y = int(df[df['image_name']==f'{image_name}.png']['corner_y'])
    corner_x = int(df[df['image_name']==f'{image_name}.png']['corner_x'])

    x_new = bbox_x + int(corner_x)
    y_new = bbox_y + int(corner_y)

    return x_new, y_new, corner_x, corner_y


def run_test_inference():

    model = set_up_computation_device()
    
    pred_boxes = []
    pred_scores = []

    # directory where all the images are present
    DIR_TEST = os.path.join(TRAIN_TEST_VALID_DIR, 'test')
    test_images = glob(os.path.join(DIR_TEST, 'block*.png'))
    print(f"Test instances: {len(test_images)}")

    preds = []
    target = []    

    for i in range(len(test_images)):

        # get the image file name for saving output later on
        image_name = test_images[i].split('\\')[-1].split('.')[0]

        print(image_name)

        image = cv2.imread(test_images[i])
        orig_image = image.copy()
        # BGR to RGB
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(float)
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float).cuda()
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            outputs = model(image)
        
        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        
        pred_boxes_single = []
        pred_scores_single = []
        pred_classes_single = []

        # carry further only if there are detected boxes
        if len(outputs[0]['boxes']) != 0:
            

            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
    
            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            labels = outputs[0]['labels'].data.numpy()[scores >= detection_threshold]
            scores = scores[scores >= detection_threshold]

            draw_boxes = boxes.copy()

            
            labels_list = labels.tolist()
            # get all the predicited class names - need to filter this according to scores
            pred_classes = [CLASSES[i] for i in labels_list]
            
            for n, i in enumerate(boxes):
                pred_boxes.append([i.tolist(), pred_classes[n]])
                pred_boxes_single.append(i.tolist())
                
            for i in scores:
                pred_scores.append(i.tolist())
                pred_scores_single.append(i.tolist())


            pred_classes_single = labels_list
    
            
            # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):

                cv2.rectangle(orig_image,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (0, 0, 255), 2)
                

                cv2.putText(orig_image, f'{pred_classes[j]}', 
                            (int(box[0]), int(box[1]-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
                            2, lineType=cv2.LINE_AA)
                
                # add to pandas dataframe - middle point of bounding box
                # image name, top left corner pixel x and y, pixel x, pixel y, pixel x and y adjusted, lat, lon, 


                out = os.path.join(OUT_DIR, f'predictions/{map_name}_test')
                if not os.path.exists(out):
                # Create a new directory because it does not exist
                    os.makedirs(out)

                cv2.imwrite(os.path.join(out, f"{image_name}.png"), orig_image)

                gt_boxes_single, gt_classes_single = get_gt_boxes_single(DIR_TEST, image_name)

                preds.append(dict(boxes=torch.as_tensor(pred_boxes_single, dtype=torch.float32), scores=torch.as_tensor(pred_scores_single, dtype=torch.float32), labels=torch.as_tensor(pred_classes_single, dtype=torch.int64)))
                target.append(dict(boxes=torch.as_tensor(gt_boxes_single, dtype=torch.float32), labels=torch.as_tensor(gt_classes_single, dtype=torch.int64)))

        print('-'*50)


    gt_boxes = get_gt_boxes(DIR_TEST)
    tp, fp, fn = get_results(gt_boxes, pred_boxes, iou_thr)

    # get results for conifers only
    gt_boxes_con = [i for i in gt_boxes if i[1] == 'conifer']
    pred_boxes_con = [i for i in pred_boxes if i[1] == 'conifer']
    tpc, fpc, fnc = get_results(gt_boxes_con, pred_boxes_con, iou_thr)

    # get results for bl only
    gt_boxes_bl = [i for i in gt_boxes if i[1] == 'tree']
    pred_boxes_bl = [i for i in pred_boxes if i[1] == 'tree']
    tpbl, fpbl, fnbl = get_results(gt_boxes_bl, pred_boxes_bl, iou_thr)


    recall = tp/(tp+fp)
    precision = tp/(tp+fn)

    print('tp =', tp)
    print('fp =', fp)
    print('fn =', fn)

    print('tp conifer =', tpc)
    print('fp conifer =', fpc)
    print('fn conifer =', fnc)

    print('tp broadleaf =', tpbl)
    print('fp broadleaf =', fpbl)
    print('fn broadleaf =', fnbl)


    metric = MeanAveragePrecision()
    metric.update(preds, target)
    from pprint import pprint
    pprint(metric.compute())


    print('recall = ', recall)
    print('precision = ', precision)

    print('TEST PREDICTIONS COMPLETE')

if __name__ == '__main__':
    run_test_inference()