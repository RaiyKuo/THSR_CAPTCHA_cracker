# THSR_CAPTCHA_cracker
Instruction how to execute .py file

Operating System: Windows 10
Python versionL:  3.6

===Preprocess.py===
Read original images in "img/" directory then pre-processed and save at "preprocessed_img/" directory

===Train.py=====
For training the model. It will read the images from the "preprocessed_img/" directory and corresponding labels csv files in "label/" directory.
Will save model at "model/" directory after finished.

===test_varify.py===
Read trained model from directory "model/" and predict the results of test images.
The test images are storaged in "preprocessed_img/" directory, and the corresponding labels are in "testing.csv" file in directory "label/"
Will comapre the prediction results and calculate the success rate.
Will output those incorrect preditions with the anwser from labels for comparison.

===demo.py===
Read trained model from directory "model\" and predict results of the images in "demo\preprocessed_img\".


ATTENTION:  this repository is for course project purpose only.
