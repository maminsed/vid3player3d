#!/bin/bash
GVHMR_ROOT_DIR="../GVHMR/outputs/demo"
TENNISPROJECT_ROOT_DIR="../TennisProject/res_smaller"
mkdir -p logs/
echo "STARTED OPERATION $(date)" &>> logs/mylogs.txt
count=0

for DIR in "$GVHMR_ROOT_DIR"/*scene*/; do
    DIR="../GVHMR/outputs/demo/0-Adrian_Mannarino_vs._Jiri_Lehecka_reencoded-scene004-000/"
    FILE_NAME="$(basename "$DIR")"
    printf "\n\n" &>> logs/mylogs.txt
    echo "Started FILE_NAME $FILE_NAME at $(date)" &>> logs/mylogs.txt
    TP_FILE_NAME=$(echo "$FILE_NAME" | sed 's/^[^-]*-//; s/-scene.*$/.csv/')
    echo "TP_FILE_NAME: $TP_FILE_NAME" &>> logs/mylogs.txt
    mkdir "./input/$FILE_NAME"
    cp "${DIR}${FILE_NAME}"_amass.pkl ./input/$FILE_NAME/.
    cp "$TENNISPROJECT_ROOT_DIR"/$TP_FILE_NAME ./input/$FILE_NAME/.

    python uhc/utils/convert_amass_isaac_correct_ground.py --amass_data ./input/"$FILE_NAME"/"$FILE_NAME"_amass.pkl --out_dir data/motion_lib/updated_"$FILE_NAME" --tennisproject_data ./input/"$FILE_NAME"/"$TP_FILE_NAME" --num_motion_libs 1 --tp_xy_units meters --tp_xy_smoothing 7 --trim_frames 30 --tp_pixel_to_meters 0.00503 --disable_xy_correction &>> logs/pythonlogs.txt

    if [ $? -ne 0 ]; then
        echo "Failed FIle $FILE_NAME at $(date)" &>> logs/mylogs.txt
        echo $FILE_NAME &>> logs/processed.txt
    else
        echo "Ended FIle $FILE_NAME at $(date)" &>> logs/mylogs.txt
    fi

    rm -rf "./input/$FILE_NAME"
    ((count++))
    break
done
echo "Finished operation after processing $count file(s) at $(date)" &>> logs/mylogs.txt
echo "Finished operation after processing $count file(s) at $(date)"


# python uhc/utils/convert_amass_isaac_correct_ground.py --amass_data input/0-Adrian_Mannarino_vs._Jiri_Lehecka_reencoded-scene000-002/0-Adrian_Mannarino_vs._Jiri_Lehecka_reencoded-scene000-002_amass.pkl --out_dir data/motion_lib/test-0-Adrian_Mannarino_vs._Jiri_Lehecka_reencoded-scene000-002 --tennisproject_data input/0-Adrian_Mannarino_vs._Jiri_Lehecka_reencoded-scene000-002/Adrian_Mannarino_vs._Jiri_Lehecka_reencoded.csv --num_motion_libs 1
