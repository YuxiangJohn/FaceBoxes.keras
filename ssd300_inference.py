from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
#from matplotlib import pyplot as plt
import cv2
from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

# Set the image size.
img_height = 300
img_width = 300


# TODO: Set the path to the `.h5` file of the model to be loaded.
model_path = 'saved_model/ssd300_pascal_07+12_epoch-99_loss-3.8301_val_loss-4.1186_single.h5'

# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

K.clear_session() # Clear previous models from memory.

model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               'DecodeDetections': DecodeDetections,
                                               'compute_loss': ssd_loss.compute_loss})
                                               
                                               
orig_images = [] # Store the images here.
input_images = [] # Store resized versions of the images here.

# We'll only load one image in this example.
img_path = 'examples/fish_bike.jpg'

orig_images.append(imread(img_path))
img = image.load_img(img_path, target_size=(img_height, img_width))
img = image.img_to_array(img) 
input_images.append(img)
input_images = np.array(input_images) 
#print(img.shape)
input_images = np.array(input_images)


y_pred = model.predict(input_images)  

confidence_threshold = 0.5
y_pred_decoded = decode_detections(y_pred,
                                   confidence_thresh=0.5,
                                   iou_threshold=0.4,
                                   top_k=200,
                                   normalize_coords=True,
                                   img_height=img_height,
                                   img_width=img_width)
#y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)
import cv2
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_decoded[0])                                   
FONT = cv2.FONT_HERSHEY_SIMPLEX
for box in y_pred_decoded[0]:
    xmin =int(round( box[2] * orig_images[0].shape[1] / img_width))
    ymin = int(round(box[3] * orig_images[0].shape[0] / img_height))
    xmax = int(round(box[4] * orig_images[0].shape[1] / img_width))
    ymax = int(round(box[5] * orig_images[0].shape[0] / img_height))
    #color = colors[int(box[0])]
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    cv2.rectangle(orig_images[0],(xmin,ymin),(xmax,ymax),(0,255,0),3)
    cv2.putText(orig_images[0], label, (xmin,ymin),
                            FONT, 2, (255, 255, 255), 2, cv2.LINE_AA)

cv2.imwrite("./result1.jpg", orig_images[0])
        