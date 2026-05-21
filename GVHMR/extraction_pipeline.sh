#!/bin/bash

mkdir -p logs
# GVHMR data extraction
DIR="../TennisProject/res_smaller"
echo "STARTED OPERATION $(date)" &>> logs/mylogs.txt
count=0


for FILE in "$DIR"/*.mp4; do
    printf "\n\n" &>> logs/mylogs.txt
    FILE_NAME="${FILE##*/}"
    
    # START=$(expr index "$FILE_NAME" "-")
    # FILE_NAME=${FILE_NAME:$START}
    # CSV_FILE_NAME=${FILE_NAME/.mp4/.csv}
    # CSV_FILE_NAME="$DIR"/$CSV_FILE_NAME

    cp "$FILE" inputs/demo/.
    echo "Started FIle $FILE_NAME at $(date)" &>> logs/mylogs.txt
    python -u tools/demo/demo_amass.py --video "inputs/demo/$FILE_NAME" --save_amass &>> logs/pythonlogs.txt
    
    if [ $? -ne 0 ]; then
        echo "Failed FIle $FILE_NAME at $(date)" &>> logs/mylogs.txt
    else
        echo "Ended FIle $FILE_NAME at $(date)" &>> logs/mylogs.txt
    fi

    rm "inputs/demo/$FILE_NAME"
    
    echo "$FILE_NAME" >> logs/processed.txt
    count=$((count+1))
    if [ "$count" -gt 50 ]; then
        break;
    fi
done
echo "Finished operation after processing $count file(s) at $(date)" &>> logs/mylogs.txt
