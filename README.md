# fr-trees-train

This code is based on the tutorial at https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/ by Sovit Ranjan Rath

This code is used to train a faster-rcnn model to detect tree symbols on historic OS map sheets.

An optional part of these is to create training data using template matching.

Data is split into training validation and test. Training and validation data are used in creating the model. Test data is used for looking at how well the model performs.

How to use:

**Before starting:** create a new conda environment and install the required packages, in environment.yml

1. ``config.py`` contains all the settings required and should be adjusted before running any code.
- ``BATCH_SIZE`` the batch size to use when training the model
- ``RESIZE_TO`` size to resize image to for training
- ``NUM_EPOCHS`` number epochs to train the model for
- ``template_image`` the path to the map image to use in template matching to generate training data.
- ``template_locs`` A list of lists of the format [[ymin, ymax, xmin, xmax], [ymin, ymax, xmin, xmax] ..] where ymin, ymax, xmin, xmax are the corners of the bounding box of the template in the template image. As many as needed can be supplied.
- ``xml_template_loc`` The path to the directory where the xml template is stored (used for creating xml files from the template matching)
- ``OUT_DIR`` is the name of the directory to store results
- ``image_dir`` path to directory that stores images for template matching
- ``TRAIN_TEST_VALID_DIR`` where training, validation and test data is stored
- ``CLASSES`` is a list of classes the model should detect (same as the model was trained with)
- ``detection threshold`` determines the score below which detections will be discarded.
- ``iou_thr`` threshold for IOU, used for getting confusion matrix on test data
- ``model_path`` is the path to the model weights
- ``scale`` is the scale of the maps that trees are being identified on. '500' is 1:500, '2500' is 1:2500 etc.
- ``city`` is the city of the maps being used
- ``generate_imgs`` should be True of False. If True, images will be generated showing the bounding boxes around idnetfied objects.
- ``slice_height`` is the height of 'patches' when a whole image is split up (in pixels)
- ``slice_width`` is the width of 'patches' when a whole image is split up (in pixels)
- ``y_overlap`` is the vertical overlap (in pixels) for pathces when a whole image is split up
- ``x_overlap`` is the horizontal overlap (in pixels) for pathces when a whole image is split up


2. Create training data using ``create_training_data.py``
   (call also be done by running ``split_image.py``, ``run_template_matching.py`` and ``create_train_valid_test.py`` individually)

   There are 2 command line arguments that can be used with ``create_training_data.py``: ``-b`` for using batch processing and ``-n`` to supply to number of batch jobs to be created.

   This is currently set up to be run on JASMIN, so should be adjusted as needed.

   Alternatively, all training data (with appropriate xml files - VOC XML format), can be provided in the ``TEST_TRAIN_VALID_DIR`` and then ``create_train_valid_test.py`` can be run to split into the three groups.

   NOTE: template matching only detects and creates xml files with 'tree' (broadleaf) symbols, not conifers. Conifer xml template is provided but is not used.

3. Run ``engine.py`` with the path to directory for train and valid data. The path must be provided as a command line argument using ``-p``. This is becuase of the use case for this project, but this should be the same as ``TRAIN_TEST_VALID_DIR``. 

   This will train the model for the amount of epochs specified in ``config.py``

4. Once the model has been trained, ensure ``model_path`` in ``config.py`` is correct for the model weights and run ``inference.py`` to run the model over the test data.
   
   The results that are output are:

   - overall recall
   - overall precision 
   - overall true positives
   - overall false positives
   - overall false negatives
   - conifer true positives
   - conifer false positives
   - conifer false negatives
   - tree (broadleaf) true positives
   - tree (broadleaf) false positives
   - tree (broadleaf) false negatives
   - statistics provided by https://torchmetrics.readthedocs.io/en/stable/detection/mean_average_precision.html 


