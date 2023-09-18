#!/usr/bin/env python

import matplotlib.pyplot as plt
from skimage import io
from skimage.feature import match_template
from skimage.feature import peak_local_max
import matplotlib.ticker as ticker
from jinja2 import Environment, FileSystemLoader
from config import image_dir, template_image, template_locs, xml_template_loc, TRAIN_TEST_VALID_DIR
from skimage.color import rgb2gray
import os
from glob import glob
import shutil

def do_template_matching(rects, bboxes, template, im):

    resulting_image = match_template(im, template)

    template_height, template_width = template.shape

    for y, x in peak_local_max(resulting_image, threshold_abs=0.6):
        rect = plt.Rectangle((x, y), template_width, template_height,
                          color='r', fc='none')
        rects.append(rect)
        xmax = x + template_width
        ymax = y + template_height

        bboxes.append({'ymin': y, 'xmin': x, 'xmax': xmax, 'ymax':ymax, 'height': template_height, 'width': template_width})
    
    return rects, bboxes

def create_xml(fname, bboxes):
    # bboxes should be list of dicts with xmin, xmax, ymin, ymax
    environment = Environment(loader=FileSystemLoader(xml_template_loc))
    template = environment.get_template("template.xml")
    
    content = template.render(name=fname, object_list=bboxes)
    
    return content

def get_templates():
    # for geotiff
    im = io.imread(template_image)
    grey_im = rgb2gray(im)
    templates = []

    for i in template_locs:
        templates.append(grey_im[i[0]:i[1], i[2]:i[3]])

    return templates

def run_template_and_xml(path):

    print('opening block image')
    im = io.imread(path)
    grey_im = rgb2gray(im)
    print('opened block image')

    fname = path.split('\\')[-1]

    # create directory if it doesn't exist
    # copy block image to train_test_valid directory
    if not os.path.exists(TRAIN_TEST_VALID_DIR):
        os.makedirs(TRAIN_TEST_VALID_DIR)
    shutil.copyfile(path, os.path.join(TRAIN_TEST_VALID_DIR, f'{fname}'))

    
    # get n from image name
    n = path.split('\\')[-1].split('.')[0][5:]
    
    rects = []
    bboxes = []

    print('Doing template matching')
    for template in get_templates():
        rects, bboxes = do_template_matching(rects, bboxes, template, grey_im)
        
    print('template matching complete')
        
        # uncomment if only wanting template matching on images where trees are identified
        # if len(bboxes) < 1:
        #     continue
    
    print('plotting tree boxes')    
    # plot rectangles
    fig = plt.figure(1, figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.NullLocator())
        
    for rect in rects:
        ax.add_patch(rect)
        
    ax.imshow(im)

    # save image
    plt.savefig(os.path.join(TRAIN_TEST_VALID_DIR,f'block{n}_templates.jpg'), dpi=300, bbox_inches='tight', pad_inches=0)
    print('plotted and save tree boxes')
        
    print('creating xml')
    # create xml file
    content = create_xml(f'block{n}.png', bboxes)
        
    # save xml
    xml_fname = os.path.join(TRAIN_TEST_VALID_DIR,f'block{n}_templates.xml')
    with open(xml_fname, mode="w") as file:
        file.write(content)
        
    print('created and saved xml')

def temp_match():
    
    paths = glob(os.path.join(image_dir, '*.png'))

    print(f'Running template matching and xml creation on {len(paths)} images')

    for p in paths:
        run_template_and_xml(p)
        print(f'Complete for image: {p}')


if __name__ == '__main__':
    temp_match()
