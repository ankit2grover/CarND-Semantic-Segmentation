import os.path
import tensorflow as tf
import helper
import project_tests as tests
import warnings
from distutils.version import LooseVersion

## Check tensorfow version and make sure that it should be greater than 1.0
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using: {}'.format(tf.__version__)
print('Tensorflow version: {}'.format(tf.__version__))

## Check tensorflow gpu device
if not tf.test.gpu_device_name():
  warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
  print('Default GPU device{}'.format(tf.test.gpu_device_name()))
  
def load_vgg(sess, vgg_path):
  """
  Load Pretrained VGG Model into TensorFlow.
  :param sess: TensorFlow Session
  :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
  :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
  """
  vgg_tag = 'vgg16'
  vgg_input_tensor_name = 'image_input:0'
  vgg_keep_prob_tensor_name = 'keep_prob:0'
  vgg_layer3_out_tensor_name = 'layer3_out:0'
  vgg_layer4_out_tensor_name = 'layer4_out:0'
  vgg_layer7_out_tensor_name = 'layer7_out:0'
  tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
  graph = tf.get_default_graph()
  print(tf.trainable_variables())
  input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
  prob_tensor = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
  layer3_tensor = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
  layer4_tensor = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
  layer7_tensor = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
  
  return input_tensor, prob_tensor, layer3_tensor, layer4_tensor, layer7_tensor

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
  """
  Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
  :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
  :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
  :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
  :param num_classes: Number of classes to classify
  :return: The Tensor for the last layer of output
  """
  conv_1x1_7 = tf.layers.conv2d(vgg_layer7_out, num_classes, (1,1), padding='same', 
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
  ## Upsample deconvolution x2
  first_upsample_7x2 = tf.layers.conv2d_transpose(conv_1x1_7, num_classes, (4,4), strides=(2,2), 
                                      padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
  
  conv_1x1_4 = tf.layers.conv2d(vgg_layer4_out, num_classes, (1,1), padding='same', 
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
  
  ## First skip layer
  first_skip_7_4 = tf.add(first_upsample_7x2, conv_1x1_4, name='first_skip')
  
  ## Upsample deconvolution x2
  second_upsample_7x2 = tf.layers.conv2d_transpose(first_skip_7_4, num_classes, (4,4), strides=(2,2), 
                                      padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
  
  conv_1x1_3 = tf.layers.conv2d(vgg_layer3_out, num_classes, (1,1), padding='same', 
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
  
  ## Second skip layer
  second_skip_7_3 = tf.add(second_upsample_7x2, conv_1x1_3, name='second_skip')
  
  ## Upsample deconvolution x8
  third_upsample_7x8 = tf.layers.conv2d_transpose(second_skip_7_3, num_classes, (16,16), strides=(8,8), 
                                      padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
  return third_upsample_7x8

tests.test_layers(layers)
  
  
def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
  
tests.test_optimize(optimize)
  
def evaluate(sess, nn_last_layer, correct_label, input_image, get_batches_fn, num_classes, batch_size, keep_prob, logits, reshape_labels):
  iou,conf_mat = tf.metrics.mean_iou( predictions=logits, labels = reshape_labels, num_classes=num_classes,name="iou")
  #iou_summary = tf.summary.scalar("IoU", iou)
  total_iou = 0;
  for images_batch, labels_batch in get_batches_fn(batch_size):
    train_iou = sess.run(iou, feed_dict={input_image:images_batch, correct_label: labels_batch, keep_prob:1})
    total_iou += train_iou * batch_size;
    print ("Total IOU: {} and mean IOU{}".format(total_iou, train_iou))
  
  total_iou = total_iou/289
  print ("Mean IOU value {}".format(total_iou))
  return total_iou

  
  
def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
  """
  Train neural network and print out the loss during training.
  :param sess: TF Session
  :param epochs: Number of epochs
  :param batch_size: Batch size
  :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
  :param train_op: TF Operation to train the neural network
  :param cross_entropy_loss: TF Tensor for the amount of loss
  :param input_image: TF Placeholder for input images
  :param correct_label: TF Placeholder for label images
  :param keep_prob: TF Placeholder for dropout keep probability
  :param learning_rate: TF Placeholder for learning rate
  """
  # Initialize all the global variables and local variables for IOU computation
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  for i in range(epochs):
    counter = 0;
    
    losses = 0
    j = 0;
    for images_batch, labels_batch in get_batches_fn(batch_size):
      _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image:images_batch, correct_label: labels_batch, keep_prob:0.9})
      print("Batch: {}".format(j))
      #print ("Iteration :{} and loss is {:3f}".format(j, loss))
      ++j
      losses += loss
      
    print("Training finished")
    print ("Epoch {} ....".format(i))
    print ("Total loss is {:3f}".format(losses * batch_size/289))
    #evaluate(sess, nn_last_layer, correct_label, input_image, get_batches_fn, num_classes, batch_size, keep_prob, logits, reshape_labels)
      
  
tests.test_train_nn(train_nn)
  
## Run the model
def run():
  num_classes = 2
  image_shape = (160, 576)
  data_dir = './data'
  runs_dir = './run'
  EPOCHS_NUM = 5
  BATCH_SIZE = 2
  learning_rate = .001
  correct_label = tf.placeholder(tf.int32, shape=[None, image_shape[0], image_shape[1], num_classes], name="correct_labels")
  #input_image = tf.placeholder(tf.float32, shape=[None, image_shape[0], image_shape[1], 3], name="input_images")
  
  ## Download VGG model
  helper.maybe_download_pretrained_vgg(data_dir)
  
  # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
  # You'll need a GPU with at least 10 teraFLOPS to train on.
  #  https://www.cityscapes-dataset.com/
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    # Path to vgg model.
    vgg_path = os.path.join(data_dir, 'vgg')
    get_batches_fn = helper.gen_batch_function(os.path.join('data_road', 'training'), image_shape)
    # OPTIONAL: Augment Images for better results
    #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

    # TODO: Build NN using load_vgg, layers, and optimize function
    input_tensor, prob_tensor, layer3_tensor, layer4_tensor, layer7_tensor = load_vgg(sess, vgg_path)
    nn_last_layer = layers(layer3_tensor, layer4_tensor, layer7_tensor, num_classes)
    logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)
    
    # TODO: Train NN using the train_nn function
    train_nn(sess, EPOCHS_NUM, BATCH_SIZE, get_batches_fn, 
             train_op, cross_entropy_loss, input_tensor, correct_label, prob_tensor, learning_rate)

    # TODO: Save inference data using helper.save_inference_samples
    helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, prob_tensor, input_tensor)

    # OPTIONAL: Apply the trained model to a video
  
if __name__ == '__main__':
    run()
  
  