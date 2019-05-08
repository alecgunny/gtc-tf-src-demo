import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import model_config_pb2

import common

import multiprocessing as mp
import argparse
import os


def load_stats(stats_path, input_shape):
  iterator = tf.io.tf_record_iterator(stats_path)
  features = {
    'mean': tf.FixedLenFeature(input_shape, tf.float32),
    'var': tf.FixedLenFeature(input_shape, tf.float32)}
  parsed = tf.parse_single_example(next(iterator), features)
  return parsed['mean'], tf.sqrt(parsed['var'])


def serving_input_receiver_fn(stats=None, spec_shape=None, eps=common._EPS):
  if stats is not None:
    shift, scale = load_stats(stats, spec_shape)

  audio_input_placeholder = tf.placeholder(
    dtype=tf.float32,
    shape=[None, common._SAMPLE_RATE],
    name='audio_input')
  receiver_tensors = {'audio_input': audio_input_placeholder}

  spectrogram = common.make_spectrogram(audio_input_placeholder)
  if stats is not None:
    spectrogram = (spectrogram - shift) / (scale + eps)

  spectrogram = tf.expand_dims(spectrogram, axis=-1)
  features = {'input_1': spectrogram}
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def parse_fn(
    record,
    table,
    input_shape,
    labels,
    shift=None,
    scale=None,
    eps=common._EPS):
  features = {
    'spec': tf.FixedLenFeature(input_shape, tf.float32),
    'label': tf.FixedLenFeature((), tf.string, default_value="")}
  parsed = tf.parse_single_example(record, features)

  # normalize spectrogrm
  spec = parsed['spec']
  if shift is not None:
    spec -= shift
  if scale is not None:
    spec /=  scale + eps
  spec = tf.expand_dims(spec, axis=2) # add channel dimension, T x F x 1

  label = tf.string_split([parsed['label']], delimiter="/").values[-2:-1]
  label = table.lookup(label)[0]
  label = tf.one_hot(label, len(labels)+1)
  return spec, label


def input_fn(
    dataset_path,
    labels,
    batch_size,
    num_epochs,
    input_shape,
    stats=None,
    eps=common._EPS,
    buffer_size=50000):
  dataset = tf.data.TFRecordDataset([dataset_path])
  table = tf.contrib.lookup.index_table_from_tensor(
    mapping=tf.constant(labels),
    num_oov_buckets=1)

  shift, scale = None, None
  if stats is not None:
    shift, scale = load_stats(stats, input_shape)

  dataset = dataset.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size, num_epochs))
  dataset = dataset.apply(
    tf.data.experimental.map_and_batch(
      map_func=lambda record: parse_fn(
        record,
        table,
        input_shape,
        labels,
        shift=shift,
        scale=scale,
        eps=eps),
      batch_size=batch_size,
      num_parallel_calls=mp.cpu_count()))
  dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return dataset


def main(FLAGS):
  labels = common.read_labels(FLAGS.labels)
  labels = labels[:20]

  # build and compile a Keras ResNet model
  input_shape = tuple(FLAGS.input_shape) + (1,)
  model = tf.keras.applications.ResNet50(
    input_shape=input_shape,
    classes=len(labels)+1,
    weights=None)
  optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
  model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy'])

  print('Training for {} steps'.format(FLAGS.max_steps))
  print(model.summary())

  # convert this model to a TF Estimator
  config = tf.estimator.RunConfig(
    save_summary_steps=FLAGS.log_steps,
    save_checkpoints_secs=FLAGS.eval_throttle_secs,
    log_step_count_steps=FLAGS.log_steps,
    tf_random_seed=0,
    model_dir=FLAGS.tensorboard_dir,
    train_distribute=tf.contrib.distribute.MirroredStrategy(
      num_gpus=FLAGS.num_gpus))
  estimator = tf.keras.estimator.model_to_estimator(model, config=config)

  # train our estimator with data from our input_fn
  train_spec = tf.estimator.TrainSpec(
    input_fn=lambda : input_fn(
      FLAGS.train_data,
      labels,
      FLAGS.batch_size,
      FLAGS.num_epochs,
      FLAGS.input_shape,
      stats=FLAGS.pixel_wise_stats),
    max_steps=FLAGS.max_steps) 
  eval_spec = tf.estimator.EvalSpec(
    input_fn=lambda : input_fn(
      FLAGS.valid_data,
      labels,
      FLAGS.batch_size*8,
      1,
      FLAGS.input_shape,
      stats=FLAGS.pixel_wise_stats),
    throttle_secs=FLAGS.eval_throttle_secs)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

  # export our model
  export_base = os.path.join(
    FLAGS.model_store_dir, FLAGS.model_name, str(FLAGS.model_version))
  tf.io.gfile.makedirs(export_base)

  # `export_saved_model` creates a timestamped directory by default
  # need to change it to fit format expected by TRTIS
  export_timestamp = estimator.export_saved_model(
    export_base,
    lambda : serving_input_receiver_fn(
      stats=FLAGS.pixel_wise_stats, spec_shape=FLAGS.input_shape))

  export_dir = os.path.join(export_base, 'model.savedmodel')
  print('Exporting saved_model to {}'.format(export_dir))

  if FLAGS.use_trt:
    print("Accelerating graph for inference with TensorRT")
    trt.create_inference_graph(
      input_graph_def=None,
      outputs=None,
      input_saved_model_dir=export_timestamp,
      input_saved_model_tags=["serve"],
      output_saved_model_dir=export_dir,
      max_batch_size=FLAGS.max_batch_size,
      max_workspace_size_bytes=1<<25,
      precision_mode=FLAGS.trt_precision)
    tf.io.gfile.rmtree(export_timestamp)
  else:
    tf.io.gfile.move(export_timestamp, export_dir)

  # export config.pbtxt and labels.txt for trtis model store
  model_config = model_config_pb2.ModelConfig()
  model_config.name = FLAGS.model_name
  model_config.max_batch_size = FLAGS.max_batch_size
  model_config.platform = 'tensorflow_savedmodel'

  model_input = model_config.input.add()
  model_input.name = 'audio_input'
  model_input.data_type = model_config_pb2.TYPE_FP32
  model_input.dims.append(common._SAMPLE_RATE)

  model_output = model_config.output.add()
  model_output.name = model.output.name.split("/")[0]
  model_output.data_type = model_config_pb2.TYPE_FP32
  model_output.dims.append(len(labels)+1)
  model_output.label_filename = 'labels.txt'

  instance_group = model_config.instance_group.add()
  instance_group.count = FLAGS.count

  config_export_base = os.path.join(FLAGS.model_store_dir, FLAGS.model_name)
  config_export_path = os.path.join(config_export_base, 'config.pbtxt')
  print('Exporting model config to {}'.format(config_export_path))
  with tf.io.gfile.GFile(config_export_path, 'wb') as f:
    f.write(str(model_config))

  labels_export_path = os.path.join(config_export_base, 'labels.txt')
  common.write_labels(labels+['unknown'], labels_export_path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--num_gpus',
    type=int,
    default=1)
  parser.add_argument(
    '--train_data',
    type=str,
    default='/data/train.tfrecords')
  parser.add_argument(
    '--valid_data',
    type=str,
    default='/data/valid.tfrecords')
  parser.add_argument(
    '--pixel_wise_stats',
    type=str,
    default=None)
  parser.add_argument(
    '--labels',
    type=str,
    default='/data/labels.txt')
  parser.add_argument(
    '--input_shape',
    nargs="+",
    type=int,
    default=[99, 161])

  parser.add_argument(
    '--learning_rate',
    type=float,
    default=1.25e-3)
  parser.add_argument(
    '--batch_size',
    type=int,
    default=512)
  parser.add_argument(
    '--num_epochs',
    type=int,
    default=25)

  parser.add_argument(
    '--tensorboard_dir',
    type=str,
    default=os.environ.get('TENSORBOARD'))
  parser.add_argument(
    '--model_store_dir',
    type=str,
    default=os.environ.get('MODELSTORE'))
  parser.add_argument(
    '--model_name',
    type=str,
    default='my_tf_model')
  parser.add_argument(
    '--model_version',
    type=int,
    default=0)
  parser.add_argument(
    '--max_batch_size',
    type=int,
    default=8)
  parser.add_argument(
    '--count',
    type=int,
    default=1)
  parser.add_argument(
    '--eval_throttle_secs',
    type=int,
    default=60)
  parser.add_argument(
    '--log_steps',
    type=int,
    default=2)
  parser.add_argument(
    '--use_trt',
    action='store_true')
  parser.add_argument(
    '--trt_precision',
    choices=("fp16", "fp32"),
    default="fp16")

  FLAGS = parser.parse_args()

  record_iterator = tf.io.tf_record_iterator(FLAGS.train_data)
  num_train_samples = len([record for record in record_iterator])
  effective_batch_size = FLAGS.batch_size * FLAGS.num_gpus
  FLAGS.max_steps = FLAGS.num_epochs*num_train_samples // (effective_batch_size)
  main(FLAGS)

