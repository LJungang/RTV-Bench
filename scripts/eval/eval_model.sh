# MODEL_NAME=llava-onevision-qwen2-7b-ov
# MODEL_PATH=/hpc2hdd/JH_DATA/share/xgao083/PrivateShareGroup/xgao083_NLPGroup/lijungang/Ckpts/llava-onevision-qwen2-7b-ov 

MODEL_NAME=qwen2.5-VL-7b-Instruct
MODEL_PATH=./Ckpts/Qwen2.5-VL-7B-Instruct

# # MODEL_NAME=qwen2.5-VL-72b-Instruct
# # MODEL_PATH=/hpc2hdd/JH_DATA/share/xgao083/PrivateShareGroup/xgao083_NLPGroup/lijungang/Ckpts/Qwen2.5-VL-72B-Instruct

SAMPLE_FRAMES=16

NUM_SPLITS=4

LOG_DIR="./run_log/parallel_${MODEL_NAME}"
mkdir -p $LOG_DIR

python ./scripts/data_process/split.py \
      --num_splits $NUM_SPLITS

for split_id in $(seq 1 $NUM_SPLITS); do
    CUDA_VISIBLE_DEVICES=$(( (split_id-1) % 2 ))  python ./scripts/eval/eval_local_model.py \
        --annotation "./data/RTV-Bench/splits/split_${split_id}.json" \
        --video_root "./data/RTV-Bench" \
        --output_dir "./eval_results/${MODEL_NAME}_${SAMPLE_FRAMES}f/split_${split_id}" \
        --splited_video_dir "./eval_results/splited_videos" \
        --model_type $MODEL_NAME \
        --model_path $MODEL_PATH \
        --sample_frames $SAMPLE_FRAMES \
        > "${LOG_DIR}/split_${split_id}.log" 2>"${LOG_DIR}/split_${split_id}.log.error" &   
done

wait
SAVE_PATH=./experiments/${MODEL_NAME}_${SAMPLE_FRAMES}f/
for split_id in $(seq 1 $NUM_SPLITS); do
    mkdir -p ${SAVE_PATH}
    cp ./eval_results/${MODEL_NAME}_${SAMPLE_FRAMES}f/split_${split_id}/results.json  ${SAVE_PATH}results_${split_id}.json 
    # rm -rf ${SAVE_PATH}*.json
done
wait
python ./scripts/data_process/merge.py \
    --target_dir ${SAVE_PATH} \
    --save_file_name ${MODEL_NAME}_${SAMPLE_FRAMES}f

rm ${SAVE_PATH}results*.json

echo "All tasks has benn done!"
