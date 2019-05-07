import tensorflow as tf
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
    'spec': _float_feature(spectrogram),
    'label': _bytes_feature(b"/".join(label.split(b"/")[-2:]))
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

  with open('{}/train/validation_list.txt'.format(FLAGS.dataset_path), 'r') as f:
    validation_files = f.read().split("\n")[:-1]
    validation_files = ['{}/train/audio/{}'.format(FLAGS.dataset_path, i) for i in validation_files]

  with open('{}/train/testing_list.txt'.format(FLAGS.dataset_path), 'r') as f:
    pseudo_test_files = f.read().split("\n")[:-1]
    pseudo_test_files = ['{}/train/audio/{}'.format(FLAGS.dataset_path, i) for i in pseudo_test_files]

  train_files = []
  for word in words:
    for fname in os.listdir('{}/train/audio/{}'.format(FLAGS.dataset_path, word)):
      filename = '{}/train/audio/{}/{}'.format(FLAGS.dataset_path, word, fname)
      if filename not in validation_files and filename not in pseudo_test_files:
        train_files.append(filename)

  # test_files = os.listdir('{}/test/audio'.format(FLAGS.dataset_path))

  dataset_files = {
    'train': train_files,
    'valid': validation_files,
    'ptest': pseudo_test_files,
    # 'test': test_files
  }[FLAGS.subset]


  # build a *SYMBOLIC* representation of our dataset
  # read audio files, batch them, build spectrograms
  dataset = tf.data.Dataset.from_tensor_slices(dataset_files)
  def serialize_example(fname):
    tf_string = tf.py_func(read_audio, fname, tf.string)
    return tf.reshape(tf_string, ())

#   dataset = dataset.apply(
#     tf.data.experimental.map_and_batch(
#       map_func=read_audio,
#       batch_size=FLAGS.batch_size,
#       num_parallel_calls=mp.cpu_count())
#     )
#   dataset = dataset.prefetch(None)
#   iterator = dataset.make_initializable_iterator()
#   spectrograms, labels = iterator.get_next()
  # spectrograms = common.make_spectrogram(audio)

  # now we'll actually iterate through build the spectrograms
  # then save them out to a TFRecord file
  # if we're doing the training set, we'll also save the pixel-wise mean
  # and variance across all spectrograms and save them out as numpy matrices
  filename = '{}/{}.tfrecords'.format(FLAGS.dataset_path, FLAGS.subset)
  writer = tf.io.TFRecordWriter(filename)

  sess = tf.Session()
  sess.run(iterator.initializer)

  print("Building tfrecord file {}".format(filename))
  progbar = tf.keras.utils.Progbar(len(dataset_files))
  first = True
  while True:
    try:
      # get a batch
      specs, labs = sess.run([spectrograms, labels])

      # if training set, update our tally of pixel wise stats
      if first and FLAGS.subset == 'train':
        mean, var = specs.sum(axis=0), (specs**2).sum(axis=0)
        first = False
      elif FLAGS.subset == 'train':
        mean += specs.sum(axis=0)
        var += (specs**2).sum(axis=0)

      # now loop through each spectrogram and label and add them to TFRecord
      # as an "example" containing named "features"
      for spectrogram, label in zip(specs, labs):
        feature = {
          'spec': _float_feature(spectrogram),
          'label': _bytes_feature(b"/".join(label.split(b"/")[-2:]))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
      progbar.add(len(specs))

    # dataset has been exhausted, we're done
    except tf.errors.OutOfRangeError:
      break
  writer.close()

  if FLAGS.subset == 'train':
    # average out our stats, use $\sigma$ = E[x**2] - E**2[x]
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
    '--batch_size',
    type=int,
    default=128,
    help='number of samples to process at once')
  
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

