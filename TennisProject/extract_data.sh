#!/bin/bash

INPUT_DIR="chopped_2/"
OUTPUT_DIR="test_2/"
mkdir -p $OUTPUT_DIR
echo &> logs/mylogs.txt
echo &> logs/pythonlogs.txt

echo "STARTED OPERATION $(date)" &>> logs/mylogs.txt
i=0
for FILE in "$INPUT_DIR"/*.mp4; do
    i=$((i + 1))
    printf "\n\n=================\n" &>> logs/mylogs.txt
    # if [ $i -lt 2 ]; then
    #     continue
    # fi
    basename="$(basename "$FILE" .mp4)"
    OUTPUT_FILE="$OUTPUT_DIR"$basename.mp4
    echo "Started FIle $FILE -> $OUTPUT_FILE at $(date)" &>> logs/mylogs.txt
    python -u main.py \
        --path_input_video $FILE \
        --path_output_video $OUTPUT_FILE \
        --path_bounce_model ctb_regr_bounce.cbm \
        --path_court_model model_tennis_court_det.pt \
        --save True \
        --path_ball_track_model model_best.pt &>> logs/pythonlogs.txt

    if [ $? -ne 0 ]; then
        echo "Failed at $(date)" &>> logs/mylogs.txt
    fi;
    echo "Finished FIle $FILE -> $OUTPUTFILE at $(date)" &>> logs/mylogs.txt
done

echo "FINISHED ENTIRE OPERATION $(date) with $i videos processed" &>> logs/mylogs.txt
