import tensorflow as tf
import tfrecord_helpers as tfhp

def worker(partitions, train_x, train_y, filename):
    """This method writes a .tfrecord file for a given range of indices

    Args:
        partitions ([list]): [tfrecord_ID, (lowerID, higherID)]
        train_x (2D NumPy array): training data array
        train_y (1D NumPy array or list): train labels array
        filename (string): beginning string of the name for the tfrecord
    """
    with tf.io.TFRecordWriter(filename+ str(partitions[0]) + '.tfrecord', 'GZIP') as tfwriter:
        for observation_id in range(partitions[1][0],partitions[1][1]):
            example_obj = tfhp.get_example_object(observation_id, data_x=train_x, data_y=train_y)
            tfwriter.write(example_obj.SerializeToString())
    return 
