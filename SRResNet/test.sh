python3 train.py \
--mode "test" \
--test_dir "/nas/k8s/dev/research/intern/jhmin/test_project/Datasets/PIPAL/CTD/test/setting1/" \
--ckpt_dir "./checkpoint/srresnet/super_resolution/upper_bound/best" \
--log_dir "./log/srresnet/super_resolution/upper_bound" \
--result_dir "./result/srresnet/super_resolution/upper_bound" \
--network "srresnet" \
--task "super_resolution" \
--opts "bicubic" 4.0 \
--learning_type "residual" \
--type div2k

# python3 train.py \
# --mode "test" \
# --test_dir "/nas/k8s/dev/research/intern/jhmin/test_project/Datasets/PIPAL/CTD/test/setting1/" \
# --ckpt_dir "/nas/k8s/dev/research/intern/jhmin/test_project/despeckle/SRResNet/checkpoint/srresnet/super_resolution/baseline_20230912_104210/best" \
# --log_dir "./log/srresnet/super_resolution/baseline_20230912_104210" \
# --result_dir "./result/srresnet/super_resolution/baseline_20230912_104210" \
# --network "srresnet" \
# --task "super_resolution" \
# --opts "bicubic" 4.0 \
# --learning_type "residual" \
# --type div2k \
