#!/bin/bash
INPUT_DIR="data/motion_lib_ik"
OUT_DIR="data/motion_lib_updated_asset/"
mkdir -p logs/

echo "STARTED OPERATION $(date)" &> logs/mylogs.txt
echo "" &> logs/pythonlogs.txt
echo "" &> logs/processed.txt
count=0

for DIR in "$INPUT_DIR"/*; do
    ((count++))
    FILE_NAME="$(basename "$DIR")"

    # if [ -e "$OUT_DIR/$FILE_NAME" ]; then
    #     echo "$OUT_DIR/$FILE_NAME" already exists &>> logs/mylogs.txt
    #     continue;
    # fi

    printf "\n\n==========\n" &>> logs/mylogs.txt
    printf "\n\n==========\n" &>> logs/pythonlogs.txt
    echo "Started FILE_NAME $FILE_NAME at $(date)" &>> logs/mylogs.txt
    
    if [ "$count" -gt 10 ]; then
        python -u uhc/utils/retarget_motion_lib.py \
            --input_dir "$DIR" \
            --out_dir $OUT_DIR/"$FILE_NAME" \
            --device cuda:0 \
            --cfg djokovic_published_v1 \
            &>> logs/pythonlogs.txt
    else
        python -u uhc/utils/retarget_motion_lib.py \
            --input_dir "$DIR" \
            --out_dir $OUT_DIR/"$FILE_NAME" \
            --device cuda:0 \
            --cfg djokovic_published_v1 \
            --verbose \
            &>> logs/pythonlogs.txt
    fi

    if [ $? -ne 0 ]; then
        echo "Failed FIle $FILE_NAME at $(date)" &>> logs/mylogs.txt
    else
        echo "Ended FIle $FILE_NAME at $(date)" &>> logs/mylogs.txt
        echo $FILE_NAME &>> logs/processed.txt
    fi
    
    # if [ "$count" -gt 110 ]; then
    #     break;
    # fi

done
echo "Finished operation after processing $count file(s) at $(date)" &>> logs/mylogs.txt
