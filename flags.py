import tensorflow as tf

tf.flags.DEFINE_string("train_data", "./data/train", "training data directory")
tf.flags.DEFINE_string("valid_data", "./data/valid", "valid data directory")

tf.flags.DEFINE_string("inputs_dir", "", "the dir for saving inputs data")

tf.flags.DEFINE_integer('max_sentence_len', 100, 'max sentence length')
tf.flags.DEFINE_string("word2vec_path", "data/model_skipgram_50.vec", "the word2vec data path")
tf.flags.DEFINE_bool('RESUME_TRAINING', False, 'resume training')
tf.flags.DEFINE_string('outputs_dir', 'trained_models', 'the dir for saving models and others')

tf.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
tf.flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
tf.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.flags.DEFINE_string('model_dir', "", "the model_dir")

tf.flags.DEFINE_integer('is_distributed', 0, ' ')
tf.flags.DEFINE_integer('batch_size', 250, ' ')
tf.flags.DEFINE_integer('eval_after_sec', 60, ' ')
tf.flags.DEFINE_integer('num_epochs', 2, ' ')
tf.flags.DEFINE_integer('max_steps', 1000000, "max steps for training")
tf.flags.DEFINE_float('keep_prob', 0.8, 'keep_prob')
tf.flags.DEFINE_float('lr', 1e-3, 'learning rate')
FLAGS = tf.flags.FLAGS

