#!/bin/bash
GVHMR_ROOT_DIR="../GVHMR/outputs/demo_2"
TP_ROOT_DIR="../TennisProject/res_2"
mkdir -p logs/

echo "STARTED OPERATION $(date)" &> logs/mylogs.txt
echo "" &> logs/pythonlogs.txt
echo "" &> logs/processed.txt
count=0

for DIR in "$GVHMR_ROOT_DIR"/*; do
    ((count++))
    FILE_NAME="$(basename "$DIR")"

    printf "\n\n==========\n" &>> logs/mylogs.txt
    printf "\n\n==========\n" &>> logs/pythonlogs.txt
    echo "Started FILE_NAME $FILE_NAME at $(date)" &>> logs/mylogs.txt
    TP_FILE_NAME="$TP_ROOT_DIR"/"$FILE_NAME".csv
    echo "TP_FILE_NAME: $TP_FILE_NAME GVHMR_DIR: $DIR" &>> logs/mylogs.txt
    
    python -u \
        uhc/utils/convert_amass_isaac_correct_ground.py \
        --gvhmr_dir "$DIR" \
        --tennisproject_data "$TP_FILE_NAME" \
        --out_dir data/motion_lib_2/"$FILE_NAME" \
        --verbose \
         &>> logs/pythonlogs.txt

    if [ $? -ne 0 ]; then
        echo "Failed FIle $FILE_NAME at $(date)" &>> logs/mylogs.txt
    else
        echo "Ended FIle $FILE_NAME at $(date)" &>> logs/mylogs.txt
        echo $FILE_NAME &>> logs/processed.txt
    fi
    
    if [ "$count" -gt 3 ]; then
        break;
    fi

done
echo "Finished operation after processing $count file(s) at $(date)" &>> logs/mylogs.txt
