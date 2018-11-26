from flags import FLAGS
import tensorflow as tf
from tools.dataset import init_worker_input_split, input_fn, serving_input_fn
from tools.loadw2c import load_w2v
from models.IDCNNTagger import IDCNNTagger
tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
    hparams = tf.contrib.training.HParams(
        num_epochs=FLAGS.num_epochs,
        batch_size=FLAGS.batch_size,
        keep_prob=FLAGS.keep_prob,
        max_steps=FLAGS.max_steps,
        learning_rate=FLAGS.lr,
        embedding=load_w2v(FLAGS.word2vec_path)[0],
        layers=[{'dilation': 1}, {'dilation': 1}, {'dilation': 2}],
        kernel_size=3,
        num_filters=128,
        blocks=3,
        lr=FLAGS.lr,
        clip_norm=5.,
        hidden_size=128,
        num_tags=4,
    )
    run_config = tf.estimator.RunConfig(
        log_step_count_steps=1000,
        tf_random_seed=19830610,
        model_dir=FLAGS.outputs_dir
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(
            init_worker_input_split(FLAGS.train_data),
            mode=tf.estimator.ModeKeys.TRAIN,
            num_epochs=hparams.num_epochs,
            batch_size=hparams.batch_size
        ),
        max_steps=hparams.max_steps,
        hooks=None
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(
            init_worker_input_split(FLAGS.valid_data),
            mode=tf.estimator.ModeKeys.EVAL,
            batch_size=hparams.batch_size
        ),

        exporters=[tf.estimator.LatestExporter(
            name="predict",  # the name of the folder in which the model will be exported to under export
            serving_input_receiver_fn=serving_input_fn,
            exports_to_keep=1,
            as_text=FLAGS)],
        steps=None,
        throttle_secs=FLAGS.eval_after_sec
    )




    if FLAGS.is_distributed:

        TF_CONFIG = {
            "cluster": {
                "ps": FLAGS.ps_hosts.split(","),
                "worker": FLAGS.worker_hosts.split(",")
            },
            "task": {"index": FLAGS.task_index, "job_name": FLAGS.job_name}}

        server = tf.train.Server(
            TF_CONFIG["cluster"],
            job_name=TF_CONFIG["task"]["job_name"],
            task_index=TF_CONFIG["task"]["index"]
        )
        if TF_CONFIG["task"]["job_name"] is "ps":
            server.join()

    estimator = tf.estimator.Estimator(model_fn=IDCNNTagger,
                                       params=hparams,
                                       config=run_config)

    tf.estimator.train_and_evaluate(
        estimator=estimator,
        train_spec=train_spec,
        eval_spec=eval_spec
    )

    # outs = estimator.predict(input_fn=lambda: input_fn(
    #     init_worker_input_split(FLAGS.valid_data),
    #     mode=tf.estimator.ModeKeys.EVAL,
    #     batch_size=hparams.batch_size
    # ))


if __name__ == '__main__':
    tf.app.run()
