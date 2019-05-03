#!/bin/bash

get_data(){
  SUBSET="$1"

  # if we don't have the extracted data, try to download it and then extract it
  # (kaggle cli will ignore the download if you already have the latest version)
  if ! [[ -d "$DATA_DIR/$SUBSET" ]]; then
    kaggle competitions download -c tensorflow-speech-recognition-challenge -f $SUBSET.7z -p $DATA_DIR
    7za x -o$DATA_DIR $DATA_DIR/$SUBSET.7z || { echo "7zip failed"; exit 1; }
  fi

  # remove the zipfile (due to size restrictions on internal cluster)
  if [[ -f "$DATA_DIR/$SUBSET.7z" ]]; then
    rm $DATA_DIR/$SUBSET.7z
  fi

  # preprocess the extracted data into smaller subsets
  if [[ $SUBSET == "train" ]]; then
    SUBSETS=( train valid ptest )
  else
    SUBSETS=( test )
  fi

  for s in "${SUBSETS[@]}"; do
    python /home/docker/preproc/preproc.py --dataset_path $DATA_DIR --batch_size 4 --log_every 50 --subset $s || \
      { echo "Encountered error in subset $s"; exit 1; }
  done
}

get_data train
# commenting out because of size restrictions on internal cluster
# get_data test
