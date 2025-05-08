# python3 train.py \
# --mode "train" \
# --train_continue "on" \
# --batch_size 32 \
# --data_dir "/nas/k8s/dev/research/intern/jhmin/test_project/Datasets/DIV2K/DIV2K_train_HR" \
# --test_dir "/nas/k8s/dev/research/intern/jhmin/test_project/Datasets/PIPAL/CTD/test/setting1/" \
# --ckpt_dir "./checkpoint/srresnet/super_resolution/ours_20230912_104025" \
# --log_dir "./log/srresnet/super_resolution/ours_20230912_104025" \
# --result_dir "./result/srresnet/super_resolution/ours_20230912_104025" \
# --network "srresnet" \
# --task "super_resolution" \
# --opts "bicubic" 4.0 \
# --learning_type "residual" \
# --type div2k \
# --weight_path "./weights_100_pipal/settings1.pkl" \
# --num_epoch 5000

python3 train.py \
--mode "train" \
--train_continue "on" \
--batch_size 32 \
--data_dir "/nas/k8s/dev/research/intern/jhmin/test_project/Datasets/DIV2K/DIV2K_train_HR" \
--test_dir "/nas/k8s/dev/research/intern/jhmin/test_project/Datasets/PIPAL/CTD/test/setting1/" \
--ckpt_dir "./checkpoint/srresnet/super_resolution/upper_bound/" \
--log_dir "./log/srresnet/super_resolution/upper_bound" \
--result_dir "./result/srresnet/super_resolution/upper_bound" \
--network "srresnet" \
--task "super_resolution" \
--opts "bicubic" 4.0 \
--learning_type "residual" \
--type div2k \
--num_epoch 5000