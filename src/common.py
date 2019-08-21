import tensorflow as tf


_SAMPLE_RATE = 16000
_FRAME_LENGTH = 20
_FRAME_STEP = 10
_EPS = 0.0001


def make_spectrogram(
    audio,
    sample_rate=_SAMPLE_RATE,
    frame_length=_FRAME_LENGTH,
    frame_step=_FRAME_STEP):
  frame_length = _FRAME_LENGTH * _SAMPLE_RATE // 1e3
  frame_step = _FRAME_STEP * _SAMPLE_RATE // 1e3

  stfts = tf.signal.stft(
    audio,
    frame_length=tf.cast(frame_length, tf.int32),
    frame_step=tf.cast(frame_step, tf.int32),
    fft_length=tf.cast(frame_length, tf.int32))
  magnitude_spectrograms = tf.abs(stfts)
  log_offset = 1e-6
  log_magnitude_spectrograms = tf.math.log(magnitude_spectrograms + log_offset)
  return tf.cast(log_magnitude_spectrograms, tf.float32)


def write_labels(labels, fname):
  with tf.io.gfile.GFile(fname, 'w') as f:
    f.write('\n'.join(labels))


def read_labels(fname):
  with tf.io.gfile.GFile(fname, 'r') as f:
    return f.read().split("\n")

