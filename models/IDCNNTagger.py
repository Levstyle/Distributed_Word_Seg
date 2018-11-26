import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xinit


def IDCNNTagger(features, labels, mode, params):
    sequences = features['sequence']
    keep_prob = tf.cond(tf.equal(mode, tf.estimator.ModeKeys.TRAIN), lambda: params.keep_prob, lambda: 1.0)
    sequence_lengths = tf.reduce_sum(tf.sign(sequences), axis=-1)
    sequences = tf.nn.embedding_lookup(params.embedding, sequences)
    sequences = tf.layers.conv1d(sequences, filters=params.num_filters,
                                 kernel_size=params.kernel_size, padding='SAME',
                                 kernel_initializer=xinit(), name="conv1d_0")
    transition_params = tf.get_variable("transitions", [params.num_tags, params.num_tags])
    outputs = []
    for block in range(params.blocks):
        for layer in range(len(params.layers)):
            dilation = params.layers[layer]['dilation']
            sequences = tf.layers.conv1d(sequences, filters=params.num_filters, kernel_size=params.kernel_size,
                                         padding='SAME', dilation_rate=dilation, kernel_initializer=xinit(),
                                         activation=tf.nn.relu, name="conv1d_{}".format(layer + 1),
                                         reuse=bool(block))
        outputs.append(sequences)

    sequences = tf.nn.dropout(tf.concat(outputs, axis=-1), keep_prob)

    scores = tf.layers.dense(sequences, units=params.num_tags, use_bias=True, kernel_initializer=xinit(),
                             name="output_fc")

    if mode == tf.estimator.ModeKeys.PREDICT:
        predicts, _ = tf.contrib.crf.crf_decode(scores, transition_params, sequence_lengths)

        # Convert predicted_indices back into strings
        predictions = {
            'class': predicts,
            'sequence_length': sequence_lengths
        }

        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }

        # Provide an estimator spec for `ModeKeys.PREDICT` modes.
        return tf.estimator.EstimatorSpec(mode,
                                          predictions=predictions,
                                          export_outputs=export_outputs)

    log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
        scores, labels, sequence_lengths, transition_params)
    loss = tf.reduce_mean(-log_likelihood)
    tf.summary.scalar('loss', loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Create Optimiser
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        # Create training operation
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        # Provide an estimator spec for `ModeKeys.TRAIN` modes.
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        predictions, _ = tf.contrib.crf.crf_decode(scores, transition_params, sequence_lengths)
        weights = tf.sequence_mask(lengths=sequence_lengths, maxlen=tf.shape(labels)[1], dtype=tf.float32)
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels, predictions, weights=weights)
        }

        # Provide an estimator spec for `ModeKeys.EVAL` modes.
        return tf.estimator.EstimatorSpec(mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)



