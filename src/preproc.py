import tensorflow as tf
tf.enable_eager_execution()
from kaggle import api as kaggle_api

import common

import os
import argparse
import subprocess
import multiprocessing as mp


def read_audio(fname):
  audio_binary = tf.read_file(fname)
  waveform = tf.contrib.ffmpeg.decode_audio(
    audio_binary,
    file_format='wav',
    samples_per_second=common._SAMPLE_RATE,
    channel_count=1)[:, 0]

  num_samples = tf.shape(waveform)[0]
  pad_front = (common._SAMPLE_RATE - num_samples) // 2
  pad_back = (common._SAMPLE_RATE - num_samples) - pad_front
  waveform = tf.pad(waveform, [[pad_front, pad_back]])

  spectrogram = common.make_spectrogram(waveform)
  feature = {
    'spec': _float_feature(spectrogram.numpy()),
    'label': _bytes_feature(b"/".join(fname.split(b"/")[-2:]))
  }
  example = tf.train.Example(features=tf.train.Features(feature=feature))
  return example.SerializeToString()


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(array):
  if array.ndim > 1:
    array = array.ravel()
  return tf.train.Feature(float_list=tf.train.FloatList(value=array))


def main(FLAGS):
  test_words = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
  aux_words = ["bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow"]
  full_word_list = os.listdir(os.path.join(FLAGS.dataset_path, 'train', 'audio'))
  del full_word_list[full_word_list.index('_background_noise_')]

  # our order will go:
  # 1. the 10 words to classify on the test set: test_words
  # 2. words that aren't one of the test set words but have a "regular" class representation
  # 3. words that aren't in the test set and are underrepresented in the training set: aux words
  # this will let us to more easily clip labels and train on reduced sets of labels
  words = [word for word in test_words]
  words += [word for word in full_word_list if word not in (aux_words  + test_words)]
  words += aux_words

  if FLAGS.subset in ['train', 'valid', 'ptest']:
    subset_path = os.path.join(FLAGS.dataset_path, 'train')
    audio_path = os.path.join(subset_path, 'audio')

    with tf.io.gfile.GFile(
        os.path.join(subset_path, 'validation_list.txt'), 'r') as f:
      validation_files = [
        os.path.join(audio_path, fname) for fname in f.read().split("\n")[:-1]]

    with tf.io.gfile.GFile(
        os.path.join(subset_path, 'testing_list.txt'), 'r') as f:
      pseudo_test_files = [
        os.path.join(audio_path, fname) for fname in f.read().split("\n")[:-1]]

    train_files = []
    for word in words:
      for fname in os.listdir(os.path.join(audio_path, word)):
        filename = os.path.join(audio_path, word, fname)
        if filename not in (validation_files + pseudo_test_files):
          train_files.append(filename)

    dataset_files = {
      'train': train_files,
      'valid': validation_files,
      'ptest': pseudo_test_files}[FLAGS.subset]

  else:
    dataset_files = os.listdir(
      os.path.join(FLAGS.dataset_path, FLAGS.subset, 'audio'))

  dataset = tf.data.Dataset.from_tensor_slices(dataset_files)
  def serialize_example(fname):
    tf_string = tf.py_func(read_audio, [fname], tf.string)
    return tf.reshape(tf_string, ())
  dataset = dataset.map(serialize_example, num_parallel_calls=mp.cpu_count())

  filename = os.path.join(FLAGS.dataset_path, FLAGS.subset+".tfrecords")
  print('Writing {} examples to dataset {}'.format(len(dataset_files), filename))
  writer = tf.data.experimental.TFRecordWriter(filename)
  writer.write(dataset)

  if FLAGS.subset == 'train':
    print('Computing pixel-wise statistics from training set')

    # average out our stats, use $\sigma$ = E[x**2] - E**2[x]
    dataset = tf.data.TFRecordDataset(filename)
    feature_spec = {'spec': tf.FixedLenFeature((99, 161), tf.float32)}
    dataset = dataset.map(lambda example:
      tf.parse_single_example(example, feature_spec))
    for n, example in enumerate(dataset.take(-1)):
      spec = example['spec'].numpy()
      if n == 0:
        mean = spec
        var = spec**2
      else:
        mean += spec
        var += spec**2

    mean /= len(dataset_files)
    var /= len(dataset_files)
    var -= mean**2

    writer = tf.io.TFRecordWriter('{}/stats.tfrecords'.format(FLAGS.dataset_path))
    features = {
      'mean': _float_feature(mean),
      'var': _float_feature(var)
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(example.SerializeToString())
    writer.close()

    common.write_labels(words, os.path.join(FLAGS.dataset_path, 'labels.txt'))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--dataset_path',
    type=str,
    default='/data',
    help='path to data')

  parser.add_argument(
    '--subset',
    type=str,
    choices=('train', 'test'),
    help='what subset to preprocess')

  FLAGS = parser.parse_args()

  subset_path = os.path.join(FLAGS.dataset_path, FLAGS.subset)
  if not os.path.exists(subset_path):
    print('Downloading and extracting data subset {}'.format(FLAGS.subset))
    zipfile_path = subset_path + ".7z"

    kaggle_api.competition_download_file(
      'tensorflow-speech-recognition-challenge',
      '{}.7z'.format(FLAGS.subset),
      FLAGS.dataset_path)
    subprocess.call(["7za", "x", "-o{}".format(FLAGS.dataset_path), zipfile_path])

  if os.path.exists(subset_path+".7z"):
    os.remove(subset_path+".7z")

  if FLAGS.subset == 'train':
    for subset in ['train', 'valid', 'ptest']:
      subset_path = os.path.join(FLAGS.dataset_path, subset+".tfrecords")
      if not os.path.exists(subset_path):
        FLAGS.subset = subset
        main(FLAGS)
  else:
    if not os.path.exists(subset_path+".tfrecords"):
      main(FLAGS)

