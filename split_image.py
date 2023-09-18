from skimage import io
from config import slice_height, slice_width, y_overlap, x_overlap, template_image
from config import image_dir
import matplotlib.pyplot as plt 
import pandas as pd
from skimage import color

def calculate_slice_bboxes(
    im,
    slice_height: int = slice_height,
    slice_width: int = slice_width,
    y_overlap: int = y_overlap,
    x_overlap: int = x_overlap,
) -> list[list[int]]:
    """
    Given the height and width of an image, calculates how to divide the image into
    overlapping slices according to the height and width provided. These slices are returned
    as bounding boxes in xyxy format.
    :return: a list of bounding boxes in xyxy format
    """
    try:
        image_height, image_width, _ = im.shape
    except ValueError:
        image_height, image_width = im.shape

    slice_bboxes = []
    y_max = y_min = 0
    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes

def run_split():
    img_list = []

    # image that is being split up
    img = io.imread(template_image)
    img = color.rgb2gray(img)
    
    # # for now, default is to split whole image into 512x512 patches with an overlap of 70 pixels
    slices = calculate_slice_bboxes(img)

    #get tiles and create csv with tile corner pixel numers
    i = 0

    for s in slices:
        xmin, ymin, xmax, ymax = s
        im = img[ymin:ymax, xmin:xmax]

        plt.imsave(f'{image_dir}/block{i}.png', im, cmap=plt.cm.gray, vmin=0, vmax=1)
        
        img_name = f'block{i}.png'
        img_dict = {'image_name': img_name, 'corner_x': xmin, 'corner_y': ymin}
        img_list.append(img_dict)

        i +=1

    print('number of blocks', len(slices))

    df = pd.DataFrame(img_list)
    df.to_csv(f'{image_dir}/img_coords.csv')

    return len(slices)
    

if __name__ == '__main__':
    run_split()
