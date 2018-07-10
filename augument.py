import numpy as np
import cv2
import math

def augument_brightness_image(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    rand_brightness = (np.random.uniform() * .9) + 0.05
    image_hsv[:,:,2] = image_hsv[:, :, 2]*rand_brightness
    return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)

def rotate_image(image, rotation_range = 20):
    # Rotation
    img_shape = image.shape
    angle_rot = np.random.uniform(rotation_range) - (rotation_range/2)
    rotation_M = cv2.getRotationMatrix2D((img_shape[1]/2, img_shape[0]/2), angle_rot, 1)
    img_rotate = cv2.warpAffine(image,rotation_M,(img_shape[1], img_shape[0]))
    return img_rotate

def shear_translate_image(image, shear_range = 25):
    # Shear Transformation
    img_shape = image.shape
    source_shear_pts = np.float32([[20,20], [40,20], [20, 40]])
    shear_pt1= 20 + np.random.uniform() * shear_range - (shear_range/2)
    shear_pt2 = 40 + np.random.uniform() * shear_range - (shear_range/4)
    dest_shear_pts = np.float32([[shear_pt1, 20], [shear_pt2, shear_pt1], [shear_pt1, 40]])
    
    shear_M = cv2.getAffineTransform(source_shear_pts, dest_shear_pts)
    img_shear = cv2.warpAffine(image,shear_M,(img_shape[1], img_shape[0]))
    return img_shear
    
def translate_image(image, translation_range = 40):
    # Translation
    img_shape = image.shape
    tr_x = translation_range*np.random.uniform()-translation_range/2
    tr_y = translation_range*np.random.uniform()-translation_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    img_translate = cv2.warpAffine(image,Trans_M,(img_shape[1], img_shape[0]))
    return img_translate


def transform_image(img, rotation_range = 90, shear_range = 40, translation_range = 40, brightness = True):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation
    
    '''    
    img = rotate_image(img, rotation_range)
    img = shear_translate_image(img, shear_range)
    img = translate_image(img, translation_range)
    if (brightness):
        img = augument_brightness_image(img)
    return img