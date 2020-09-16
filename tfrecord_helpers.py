import tensorflow as tf

def _byte_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def array_bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value.tobytes()))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def array_floats_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_example_object(observation_id=0, data_x=None, data_y=None):
    """ This function returns an example for a given train data point 
        and a label.

    Args:
        observation_id (int): training/testing data ID to convert to an example.
        data_x (2D NumPy array): training/testing inputs.
        data_y (1D NumPy array): training/testing label.
    """
    feature = {
        'label': _int64_feature(data_y[observation_id]),
        'image_raw': array_floats_feature(data_x[observation_id]),
  }
    example_obj = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_obj
