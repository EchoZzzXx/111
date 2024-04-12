export DATASET_PATH='/home/rjy/CARLA2023/model_test/perception/dataset'

cd ./src/perception
python3 ./train.py ${DATASET_PATH} \
    --dataset carla \
    --train-towns 2 4 5 6 \
    --val-towns 3 \
    --train-weathers 0 --val-weathers 1 \
    --model img2map_model --sched cosine --epochs 40 --warmup-epochs 5 --lr 0.0003 --batch-size 32 \
    --workers 32 --eval-metric l1_error \
    --opt adamw --opt-eps 1e-8 --weight-decay 0.05  \
    --scale 0.9 1.1 --saver-decreasing --clip-grad 10 --freeze-num -1 \
    --with-backbone-lr --backbone-lr 0.0002 \
    --multi-view --with-lidar --multi-view-input-size 3 128 128 \
    --experiment img2map_test \
    # --pretrained --initial-checkpoint 'logs/test.pth.tar'
