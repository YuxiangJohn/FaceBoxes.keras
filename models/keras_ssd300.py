'''
A Keras port of the original Caffe SSD300 network.

Copyright (C) 2018 Pierluigi Ferrari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import division
import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate, BatchNormalization
from keras.regularizers import l2
import keras.backend as K

from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_L2Normalization import L2Normalization
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

def ssd_300(image_size,
            n_classes = 1,
            mode='training',
            l2_regularization=0.0005,
            min_scale=[[32], [256], [512]],
            max_scale=None,
            scales=None,
            aspect_ratios_global=None,
            aspect_ratios_per_layer=[[1.0, 1.0, 1.0],
                                     [1.0, 1.0, 1.0],
                                     [1.0, 1.0, 1.0]],
            two_boxes_for_ar1=True,
            steps=[32, 64, 128],
            offsets=None,
            clip_boxes=False,
            variances=[0.1, 0.1, 0.2, 0.2],
            coords='centroids',
            normalize_coords=True,
            subtract_mean=[123, 117, 104],
            divide_by_stddev=None,
            swap_channels=[2, 1, 0],
            confidence_thresh=0.01,
            iou_threshold=0.45,
            top_k=200,
            nms_max_output_size=400,
            return_predictor_sizes=False):
    '''
    Build a Keras model with SSD300 architecture, see references.

    The base network is a reduced atrous VGG-16, extended by the SSD architecture,
    as described in the paper.

    Most of the arguments that this function takes are only needed for the anchor
    box layers. In case you're training the network, the parameters passed here must
    be the same as the ones used to set up `SSDBoxEncoder`. In case you're loading
    trained weights, the parameters passed here must be the same as the ones used
    to produce the trained weights.

    Some of these arguments are explained in more detail in the documentation of the
    `SSDBoxEncoder` class.

    Note: Requires Keras v2.0 or later. Currently works only with the
    TensorFlow backend (v1.0 or later).

    Arguments:
        image_size (tuple): The input image size in the format `(height, width, channels)`.
        n_classes (int): The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO.
        mode (str, optional): One of 'training', 'inference' and 'inference_fast'. In 'training' mode,
            the model outputs the raw prediction tensor, while in 'inference' and 'inference_fast' modes,
            the raw predictions are decoded into absolute coordinates and filtered via confidence thresholding,
            non-maximum suppression, and top-k filtering. The difference between latter two modes is that
            'inference' follows the exact procedure of the original Caffe implementation, while
            'inference_fast' uses a faster prediction decoding procedure.
        l2_regularization (float, optional): The L2-regularization rate. Applies to all convolutional layers.
            Set to zero to deactivate L2-regularization.
        min_scale (float, optional): The smallest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images.
        max_scale (float, optional): The largest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images. All scaling factors between the smallest and the
            largest will be linearly interpolated. Note that the second to last of the linearly interpolated
            scaling factors will actually be the scaling factor for the last predictor layer, while the last
            scaling factor is used for the second box for aspect ratio 1 in the last predictor layer
            if `two_boxes_for_ar1` is `True`.
        scales (list, optional): A list of floats containing scaling factors per convolutional predictor layer.
            This list must be one element longer than the number of predictor layers. The first `k` elements are the
            scaling factors for the `k` predictor layers, while the last element is used for the second box
            for aspect ratio 1 in the last predictor layer if `two_boxes_for_ar1` is `True`. This additional
            last scaling factor must be passed either way, even if it is not being used. If a list is passed,
            this argument overrides `min_scale` and `max_scale`. All scaling factors must be greater than zero.
        aspect_ratios_global (list, optional): The list of aspect ratios for which anchor boxes are to be
            generated. This list is valid for all prediction layers.
        aspect_ratios_per_layer (list, optional): A list containing one aspect ratio list for each prediction layer.
            This allows you to set the aspect ratios for each predictor layer individually, which is the case for the
            original SSD300 implementation. If a list is passed, it overrides `aspect_ratios_global`.
        two_boxes_for_ar1 (bool, optional): Only relevant for aspect ratio lists that contain 1. Will be ignored otherwise.
            If `True`, two anchor boxes will be generated for aspect ratio 1. The first will be generated
            using the scaling factor for the respective layer, the second one will be generated using
            geometric mean of said scaling factor and next bigger scaling factor.
        steps (list, optional): `None` or a list with as many elements as there are predictor layers. The elements can be
            either ints/floats or tuples of two ints/floats. These numbers represent for each predictor layer how many
            pixels apart the anchor box center points should be vertically and horizontally along the spatial grid over
            the image. If the list contains ints/floats, then that value will be used for both spatial dimensions.
            If the list contains tuples of two ints/floats, then they represent `(step_height, step_width)`.
            If no steps are provided, then they will be computed such that the anchor box center points will form an
            equidistant grid within the image dimensions.
        offsets (list, optional): `None` or a list with as many elements as there are predictor layers. The elements can be
            either floats or tuples of two floats. These numbers represent for each predictor layer how many
            pixels from the top and left boarders of the image the top-most and left-most anchor box center points should be
            as a fraction of `steps`. The last bit is important: The offsets are not absolute pixel values, but fractions
            of the step size specified in the `steps` argument. If the list contains floats, then that value will
            be used for both spatial dimensions. If the list contains tuples of two floats, then they represent
            `(vertical_offset, horizontal_offset)`. If no offsets are provided, then they will default to 0.5 of the step size.
        clip_boxes (bool, optional): If `True`, clips the anchor box coordinates to stay within image boundaries.
        variances (list, optional): A list of 4 floats >0. The anchor box offset for each coordinate will be divided by
            its respective variance value.
        coords (str, optional): The box coordinate format to be used internally by the model (i.e. this is not the input format
            of the ground truth labels). Can be either 'centroids' for the format `(cx, cy, w, h)` (box center coordinates, width,
            and height), 'minmax' for the format `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
        normalize_coords (bool, optional): Set to `True` if the model is supposed to use relative instead of absolute coordinates,
            i.e. if the model predicts box coordinates within [0,1] instead of absolute coordinates.
        subtract_mean (array-like, optional): `None` or an array-like object of integers or floating point values
            of any shape that is broadcast-compatible with the image shape. The elements of this array will be
            subtracted from the image pixel intensity values. For example, pass a list of three integers
            to perform per-channel mean normalization for color images.
        divide_by_stddev (array-like, optional): `None` or an array-like object of non-zero integers or
            floating point values of any shape that is broadcast-compatible with the image shape. The image pixel
            intensity values will be divided by the elements of this array. For example, pass a list
            of three integers to perform per-channel standard deviation normalization for color images.
        swap_channels (list, optional): Either `False` or a list of integers representing the desired order in which the input
            image channels should be swapped.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
            positive class in order to be considered for the non-maximum suppression stage for the respective class.
            A lower value will result in a larger part of the selection process being done by the non-maximum suppression
            stage, while a larger value will result in a larger part of the selection process happening in the confidence
            thresholding stage.
        iou_threshold (float, optional): A float in [0,1]. All boxes that have a Jaccard similarity of greater than `iou_threshold`
            with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
            to the box's confidence score.
        top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
            non-maximum suppression stage.
        nms_max_output_size (int, optional): The maximal number of predictions that will be left over after the NMS stage.
        return_predictor_sizes (bool, optional): If `True`, this function not only returns the model, but also
            a list containing the spatial dimensions of the predictor layers. This isn't strictly necessary since
            you can always get their sizes easily via the Keras API, but it's convenient and less error-prone
            to get them this way. They are only relevant for training anyway (SSDBoxEncoder needs to know the
            spatial dimensions of the predictor layers), for inference you don't need them.

    Returns:
        model: The Keras SSD300 model.
        predictor_sizes (optional): A Numpy array containing the `(height, width)` portion
            of the output tensor shape for each convolutional predictor layer. During
            training, the generator function needs this in order to transform
            the ground truth labels into tensors of identical structure as the
            output tensors of the model, which is in turn needed for the cost
            function.

    References:
        https://arxiv.org/abs/1512.02325v5
    '''

    n_predictor_layers = 3 # The number of predictor conv layers in the network is 6 for the original SSD300.
    n_classes += 1 # Account for the background class.
    l2_reg = l2_regularization # Make the internal name shorter.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################

    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError("`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers+1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(n_predictor_layers+1, len(scales)))
    else: # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1) # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    else: # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    ############################################################################
    # Define functions for the Lambda layers below.
    ############################################################################

    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]], tensor[...,swap_channels[3]]], axis=-1)

    ##### inception block #####
    def inception_module(x, para):
        #path 1
        x1 = Conv2D(32, (1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name=para+'Conv1')(x)
        #path 2
        y = MaxPooling2D(pool_size=(3, 3), strides =(1,1), padding='same', name=para+'pool')(x)
        x2 = Conv2D(32, (1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name=para+'Conv2')(y)
        #path 3
        y = Conv2D(24, (1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name=para+'Conv3_1')(x)
        x3 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name=para+'Conv1')(y)
        #path 4
        y = Conv2D(24, (1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name=para+'Conv4_1')(x)
        y = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name=para+'Conv4_2')(y)
        x4 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name=para+'Conv4_3')(y)
        x = Concatenate(axis=3, name=para+'concat')([x1, x2, x3, x4])
        return x
        
    ############################################################################
    # Build the network.
    ############################################################################

    data_input = Input(shape=(img_height, img_width, img_channels))
    #### Rapidly Digested Conv Layers ###
    x = Conv2D(24, (7, 7), strides=(4, 4), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1')(data_input)
    x = BatchNormalization(gamma_regularizer=l2(l2_reg), beta_regularizer=l2(l2_reg))(x)
    x = Concatenate(axis=3, name='concat_n_1')([x, -x])
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(x)
    
    x = Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2')(x)
    x = BatchNormalization(gamma_regularizer=l2(l2_reg), beta_regularizer=l2(l2_reg))(x)
    x = Concatenate(axis=3, name='concat_n_2')([x, -x])
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool2')(x)    
    
    #### Multiple Scale Conv Layers ####
    
    x = inception_module(x, 'inception_block1_')
    x = inception_module(x, 'inception_block2_')
    inception_output = inception_module(x, 'inception_block3_')
      #scale 0
    x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_1')(inception_output)
    Conv3_2_output = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_2')(x)
      # scale 1
    x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_1')(Conv3_2_output)
    Conv4_2_output = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_2')(x)
      # scale 2
    
    
    #### confidence predictors ####
    #inception3_conf
    inception3_conf = Conv2D(6, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='inception3_conf')(inception_output)
    inception3_conf_reshape = Reshape((-1, n_classes), name='inception3_mbox_conf_reshape')(inception3_conf)
    #print("inception3_conf: ",inception3_conf.shape)
    #print("inception3_conf_reshape: ",inception3_conf_reshape.shape)
    #Conv3_2_conf
    Conv3_2_conf = Conv2D(2, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_2_conf')(Conv3_2_output)
    Conv3_2_conf_reshape = Reshape((-1, n_classes), name='conv3_2_mbox_conf_reshape')(Conv3_2_conf)
    #Conv4_2_conf
    Conv4_2_conf = Conv2D(2, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_2_conf')(Conv4_2_output)
    Conv4_2_conf_reshape = Reshape((-1, n_classes), name='conv4_2_mbox_conf_reshape')(Conv4_2_conf)
    #concatenate conf
    mbox_conf =  Concatenate(axis=1, name='mbox_conf')([inception3_conf_reshape, Conv3_2_conf_reshape,  Conv4_2_conf_reshape])
    
    #### locations predictors ####
    #inception3_loc 
    inception3_loc = Conv2D(12, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='inception3_loc')(inception_output)
    inception3_loc_reshape = Reshape((-1, 4), name='inception3_mbox_loc_reshape')(inception3_loc)

    #Conv3_2_loc
    Conv3_2_loc = Conv2D(4, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_2_loc')(Conv3_2_output)
    Conv3_2_loc_reshape = Reshape((-1, 4), name='conv3_2_mbox_loc_reshape')(Conv3_2_loc)

    #Conv4_2_loc
    Conv4_2_loc = Conv2D(4, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_2_loc')(Conv4_2_output)
    Conv4_2_loc_reshape = Reshape((-1, 4), name='conv4_2_mbox_loc_reshape')(Conv4_2_loc)

    #concatenate loc
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([inception3_loc_reshape, Conv3_2_loc_reshape, Conv4_2_loc_reshape])
    
    #### prior boxes ####
    #inception3_priorbox
    
    inception3_2_priorbox_1 = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0],
                                             two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0], clip_boxes=clip_boxes,
                                             variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv3_2_priorbox_1')(inception3_loc)
    
    inception3_2_priorbox_2 = AnchorBoxes(img_height, img_width, this_scale=scales[0]*2, next_scale=scales[1], aspect_ratios=aspect_ratios[0],
                                             two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0], clip_boxes=clip_boxes,
                                             variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv3_2_priorbox_2')(inception3_loc)
    inception3_2_priorbox_3 = AnchorBoxes(img_height, img_width, this_scale=scales[0]*4, next_scale=scales[1], aspect_ratios=aspect_ratios[0],
                                             two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0], clip_boxes=clip_boxes,
                                             variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv3_2_priorbox_3')(inception3_loc)
    inception3_2_priorbox_1_reshape = Reshape((-1, 8), name='inception3_2_priorbox_1_reshape')(inception3_2_priorbox_1)
    inception3_2_priorbox_2_reshape = Reshape((-1, 8), name='inception3_2_priorbox_2_reshape')(inception3_2_priorbox_2)
    inception3_2_priorbox_3_reshape = Reshape((-1, 8), name='inception3_2_priorbox_3_reshape')(inception3_2_priorbox_3)

    
    #Conv3_2_priorbox                                         
    Conv3_2_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
                                             two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1], clip_boxes=clip_boxes,
                                             variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv3_2_priorbox')(Conv3_2_loc)
    Conv3_2_priorbox_reshape = Reshape((-1, 8), name='conv3_2_priorbox_reshape')(Conv3_2_priorbox)
    #Conv4_2_priorbox
    Conv4_2_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
                                             two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2], this_offsets=offsets[2], clip_boxes=clip_boxes,
                                             variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv4_2_priorbox')(Conv4_2_loc)
    Conv4_2_priorbox_reshape = Reshape((-1, 8), name='conv4_2_priorbox_reshape')(Conv4_2_priorbox)
    #concatenate priorbox
    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([inception3_2_priorbox_1_reshape, inception3_2_priorbox_2_reshape, inception3_2_priorbox_3_reshape, Conv3_2_priorbox_reshape, Conv4_2_priorbox_reshape])
    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)
    
    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_priorbox])
    
    if mode == 'training':
        model = Model(inputs=data_input, outputs=predictions)
    elif mode == 'inference':
        decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               coords=coords,
                                               normalize_coords=normalize_coords,
                                               img_height=img_height,
                                               img_width=img_width,
                                               name='decoded_predictions')(predictions)
        model = Model(inputs=data_input, outputs=decoded_predictions)
    elif mode == 'inference_fast':
        decoded_predictions = DecodeDetectionsFast(confidence_thresh=confidence_thresh,
                                                   iou_threshold=iou_threshold,
                                                   top_k=top_k,
                                                   nms_max_output_size=nms_max_output_size,
                                                   coords=coords,
                                                   normalize_coords=normalize_coords,
                                                   img_height=img_height,
                                                   img_width=img_width,
                                                   name='decoded_predictions')(predictions)
        model = Model(inputs=data_input, outputs=decoded_predictions)
    else:
        raise ValueError("`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(mode))                                    
    
                                        

    if return_predictor_sizes:
        predictor_sizes = np.array([inception3_2_conf._keras_shape[1:3],
                                     Conv3_2_conf._keras_shape[1:3],
                                     Conv4_2_conf._keras_shape[1:3]])
        return model, predictor_sizes
    else:
        return model
