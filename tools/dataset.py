import tensorflow as tf
import multiprocessing
from flags import FLAGS

def input_fn(file_names, mode=tf.estimator.ModeKeys.EVAL, num_epochs=1, batch_size=200):
    shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
    num_threads = multiprocessing.cpu_count()
    buffer_size = 2 * batch_size + 1

    files = tf.data.Dataset.list_files(file_names)
    # number_of_cpu is the value of worker.vcore in xml file
    dataset = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TextLineDataset,
        cycle_length=4 * 2))

    def parse(value):
        decoded = tf.decode_csv(
            value,
            field_delim=' ',
            record_defaults=[[0]] * FLAGS.max_sentence_len * 2)
        return tf.split(decoded, 2, axis=-1)

    dataset = dataset.map(parse, num_parallel_calls=num_threads)
    dataset = dataset.batch(batch_size)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(buffer_size)

    iterator = dataset.make_one_shot_iterator()

    sequence, labels = iterator.get_next()

    return {'sequence': sequence}, labels


def init_worker_input_split(dir_path):
    import os
    file_names = []
    for current_file_name in tf.gfile.ListDirectory(dir_path):
        file_path = os.path.join(dir_path, current_file_name)
        file_names.append(file_path)
    return file_names

def serving_input_fn():
    receiver_tensor = {
        'sequence': tf.placeholder(tf.int32, [None, None]),
    }

    features = {
        key: tensor for key, tensor in receiver_tensor.items()
    }

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)