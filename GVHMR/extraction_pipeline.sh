#!/bin/bash

mkdir -p logs
# GVHMR data extraction
INPUT_DIR="../TennisProject/res_2"
HIGHER_RES_DIR="../TennisProject/chopped_2"
echo "STARTED OPERATION $(date)" &>> logs/mylogs.txt
count=0

# echo "" &> logs/mylogs.txt
# echo "" &> logs/pythonlogs.txt
# echo "" &> logs/processed.txt

CSV_PATH=../TennisProject/res_2/Alexander_Bublik_vs._Tommy_Paul_reencoded-scene002-000.csv 

for FILE in "$INPUT_DIR"/*.mp4; do
    FILE_NAME="${FILE##*/}"

    # if [ "$FILE_NAME" != "0-Amanda_Anisimova_vs._Beatriz_Haddad_Maia_reencoded-scene018-000.mp4" ]; then
    #     echo skipped $FILE_NAME &>> logs/tmp.txt
    #     continue;
    # fi

    count=$((count+1))
    printf "\n\n" &>> logs/mylogs.txt
    if [ "$count" -le 50 ]; then
        echo skipped "$FILE_NAME" &>> logs/mylogs.txt
        continue;
    fi
    printf "\n\n\n\n====================\n" &>> logs/pythonlogs.txt
    
    # START=$(expr index "$FILE_NAME" "-")
    # FILE_NAME=${FILE_NAME:$START}
    # CSV_FILE_NAME=${FILE_NAME/.mp4/.csv}
    # CSV_FILE_NAME="$DIR"/$CSV_FILE_NAME

    basename="$(basename "$FILE" .mp4)"
    CSV_PATH="$INPUT_DIR"/"${basename#0-}".csv
    INPUT_VIDEO="$HIGHER_RES_DIR"/"${basename#0-}".mp4
    TEMP_DIR=inputs/demo/"${basename#0-}".mp4

    end_frame=$(python -c "import pandas as pd; df = pd.read_csv(\""$CSV_PATH"\"); print(df.loc[:,\"global_frame\"].iloc[-1] - 3)") &>> logs/mylogs.txt
    start_frame=$(python -c "import pandas as pd; df = pd.read_csv(\""$CSV_PATH"\"); print(df.loc[0,\"global_frame\"] + 3)") &>> logs/mylogs.txt

    echo "$(date): Started FIle_NAME: $FILE_NAME - CSV_PATH: $CSV_PATH - INPUT_VIDEO: $INPUT_VIDEO - start_frame: $start_frame - end_frame: $end_frame" &>> logs/mylogs.txt

    cp "$INPUT_VIDEO" "$TEMP_DIR"
    python -u tools/demo/demo_amass.py \
        --video "$TEMP_DIR" \
        --output_root "outputs/demo_2/" \
        --start_frame $start_frame \
        --end_frame $end_frame \
        --save_amass &>> logs/pythonlogs.txt # remove verbose!
    
    if [ $? -ne 0 ]; then
        echo "Failed FIle $FILE_NAME at $(date)" &>> logs/mylogs.txt
    else
        echo "$FILE_NAME" >> logs/processed.txt
        echo "Ended FIle $FILE_NAME at $(date)" &>> logs/mylogs.txt
    fi

    rm "$TEMP_DIR"
    
    if [ "$count" -gt 100 ]; then
        break;
    fi
done
echo "Finished operation after processing $count file(s) at $(date)" &>> logs/mylogs.txt
