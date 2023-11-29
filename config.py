import torch
import os 

BATCH_SIZE = 1 # increase / decrease according to GPU memeory
RESIZE_TO = 512 # resize the image for training and transforms
NUM_EPOCHS = 200 # number of epochs to train for

def get_map_name(map_im):
    map_name = map_im.split('\\')[-1].split('.')[0] 
    return map_name

# path to original rgb map image
template_image = '..\..\map_images\Edinburgh_1_2500\\125642410.27.tif'

# location to save model and plots
OUT_DIR = 'model/extra'
model_path = os.path.join(OUT_DIR, 'best.pth')

map_name = get_map_name(template_image)

city = 'leeds'
scale = '500'

# path to directory that stores images to run model over
image_dir = f'..\split_ims\{city}\\1_{scale}\{map_name}\greyscale'

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# used in splitting into test, train, valid
TRAIN_TEST_VALID_DIR = '..\\training_data\\extra'

# classes: 0 index is reserved for background
CLASSES = [
    'background', 'tree', 'conifer'
]

# any detection having score below this will be discarded
# currently using avg of leeds and edi
initial_detection_threshold = 0.7
final_conifer_threshold_leeds = 0.849
final_broadleaf_threshold_leeds = 0.972
final_conifer_threshold_edi = 0.975
final_broadleaf_threshold_edi = 0.984

#leeds vals
#con = 0.849 
#broadleaf = 0.972

#edi vals
#con = 0.975
#broadleaf = 0.984 

iou_thr = 0.6

# template locations in the image - [ymin, ymax, xmin, xmax]]
template_locs = [[3651,3706,14835,14889], [3668,3717,13806,13841], [3631,3694,2214,2283], [3149,3204,1208,1260]]
xml_template_loc = '.'

# image slice parameters
slice_height = 512
slice_width = 512
# generally use 70 unless edi 1:500 - then use 150
y_overlap = 150
x_overlap = 150