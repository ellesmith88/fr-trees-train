# This code is based on the tutorial at https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/ by Sovit Ranjan Rath


NOTE: template matching only detects and creates xml files with 'tree' symbols, not conifers. Conifer xml template is provided but is not used.

1. Create training data using create_training_data.py (call also be done by running split_image.py, run_template_matching.py and create_train_valid_test.py individually)
2. Run engine.py with path to directory for train and valid data
3. Run inference.py to run model over test data and get results