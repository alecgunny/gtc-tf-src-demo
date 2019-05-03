## Building the container
```
$ docker build -t $USER/tf-speech-recognition .
```

## Build necessary volumes
```
$ docker volume create modelstore tensorboard
```

## Launch tensorboard
```
$ docker run --rm -d -v tensorboard:/tensorboard -p 6006:6006 --name tensorboard \
    $USER/tf-speech-recognition tensorboard --logdir=/tensorboard --host=0.0.0.0
```

## Train the model
```
$ DATA_DIR=/path/to/data
$ docker run --rm -it -v tensorboard:/tensorboard -v modelstore:/modelstore -v $DATA_DIR:/data \
    $USER/tf-speech-recognition python main.py --num_gpus 4 \
    --train_data /data/train.tfrecords --valid_data /data/valid.tfrecords --pixel_wise_stats /data/stats.tfrecords \
    --input_shape 99 161 --labels /data/labels.txt --num_epochs 5 --model_name my_tf_model \
    --max_batch_size 8 --count 4 --use_trt --trt_precision fp16
```

## Launch TensorRT Inference Server
```
$ docker run --rm -d -v modelstore:/modelstore -p 8000-8002:8000-8002 -e NVIDIA_VISIBLE_DEVICES=0 --name trtserver \
    nvcr.io/nvidia/tensorrtserver:19.04-py3 trtserver --model-store=/modelstore
```

## Build TensorRT client container
```
docker build -t $USER/tensorrtserver:client --target=trtisclient .
```

## Run performance client
```
$ docker run --rm -it --network container:trtserver $USER/tensorrtserver:client \
    /opt/tensorrtserver/clients/bin/perf_client -m my_tf_model -u localhost:8001 -i grpc -d -l 1000 -c 8 -b 8 -p 5000
```
