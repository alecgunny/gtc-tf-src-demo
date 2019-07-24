# Sample TensorFlow Workflow
Here's a simple workflow for training a deep learning model with TensorFlow then exporting it for inference with NVIDIA's <a href=https://github.com/NVIDIA/tensorrt-inference-server>TensorRT Inference Server</a>.

## Fetch and build the dataset
For this example, we'll use some data from a Kaggle competition I worked on a while back, but feel free to use any data you see fit. To download the data, you'll first need to get a Kaggle API key by <a href=https://github.com/Kaggle/kaggle-api>following these instructions</a>. Once you have your config json saved at `/path/to/.kaggle`,
```
KAGGLE_DIR=/path/to/.kaggle
DATA_DIR=/path/to/data

docker build -t $USER/tensorflow-sample:preproc -f Dockerfile.preproc \
    github.com/alecgunny/gtc-tf-src-demo.git
docker run -v $DATA_DIR:/data -v $KAGGLE_DIR:/tmp/.kaggle -u $(id -u):$(id -g) \
    $USER/tensorflow-sample:preproc python preproc.py --data_dir /data --subset train
```

## Training the model
Once our data is ready, let's build our training container, built on top of the TensorFlow image from NVIDIA's <a href=ngc.nvidia.com>NVIDIA GPU Cloud</a> container repository.
```
docker build -t $USER/tensorflow-sample https://github.com/alecgunny/gtc-tf-src-demo.git
```
### Monitoring the model
We'll launch TensorFlow's tensorboard utility in the background to monitor the progress of our model as it trains
```
<<<<<<< HEAD
$ docker volume create tensorboard
$ docker run --rm -d -v tensorboard:/tensorboard -p 6006:6006 --name tensorboard \
    $USER/tensorflow-sample tensorboard --logdir=/tensorboard --host=0.0.0.0
=======
docker volume create tensorboard
docker run --rm -d -v tensorboard:/tensorboard -p 6006:6006 --name tensorboard \
    $USER/tf-speech-recognition tensorboard --logdir=/tensorboard --host=0.0.0.0
>>>>>>> 8770a063ec7d1681000343b804ff20bb332f40ce
```
Navigate to `localhost:6006` on your machine to keep track of model loss and accuracy.

### Train the model
Now we're all set to launch our training script! Feel free to look in `src/main.py` to see the default values used for things like learning rate, batch size, etc. Note that even though we're training our model on 2D spectrograms, we'll use a `serving_input_receiver_fn` to export a model which takes in raw one second audio signals, computes a spectrogram, then runs the model.

We also provide some flags which will tell the TensorRT Inference Server how to serve up our model once it's exported.
- `--max_batch_size` tells the inference server the maximum number of audio clips it can expect to handle at once
- `--count` is used to tell the inference server how many threads of inference to run concurrently
- `--use_trt` accelerates the model with NVIDIA's TensorRT inference library integrated natively into TensorFlow
- `--trt_precision` specifies at what precision to run inference. Using `fp16` precision can drastically improve throughput
```
$ docker volume create modelstore
$ docker run --rm -it -v tensorboard:/tensorboard -v modelstore:/modelstore -v $DATA_DIR:/data \
    $USER/tensorflow-sample python main.py --num_gpus 4 --train_data /data/train.tfrecords \
    --valid_data /data/valid.tfrecords --pixel_wise_stats /data/stats.tfrecords \
    --input_shape 99 161 --labels /data/labels.txt --num_epochs 5 --model_name my_tf_model \
    --max_batch_size 8 --count 4 --use_trt --trt_precision fp16
```

## Do stuff with the model
Now that our model as been trained and exported, let's use the TensorRT Inference Server to serve it up and expose it to the world to do cool stuff.

### Launch the inference server
Point the inference server at our model store volume, expose the appropriate ports (8000 for http, 8001 for grpc, and 8002 for metrics), and launch the TRT server locally on a single GPU.
```
docker run --rm -d -v modelstore:/modelstore -p 8000-8002:8000-8002 -e NVIDIA_VISIBLE_DEVICES=0 \
    --name trtserver nvcr.io/nvidia/tensorrtserver:19.04-py3 trtserver --model-store=/modelstore
```

### Build the client container image
Build a container image with the appropriate container libraries to make calls to the server.
```
docker build -t $USER/tensorrtserver:client --target=trtisclient \
    https://github.com/alecgunny/gtc-tf-src-demo.git
```

### Benchmark with the performance client
The TensorRT server comes with a handy client for benchmarking model throughput and latency under varying loads. As an example, let's run this performance client up to 8 concurrent calls (the concurrent calls are a proxy for the load size, and approximate a server with an average queue size of 8).
```
docker run --rm -it --network container:trtserver $USER/tensorrtserver:client \
    /opt/tensorrtserver/clients/bin/perf_client -m my_tf_model -u localhost:8001 -i grpc \
    -d -l 1000 -c 8 -b 8 -p 5000
```
