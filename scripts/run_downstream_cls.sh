images_path=/mnt/dfs/jpang12/datasets/nih_xray14/images/images
train_file_path=/mnt/dfs/ssiingh/BenchmarkArk/dataset/Xray14_train_official.txt
val_file_path=/mnt/dfs/ssiingh/BenchmarkArk/dataset/Xray14_val_official.txt
test_file_path=/mnt/dfs/ssiingh/BenchmarkArk/dataset/Xray14_test_official.txt
pretrained_model_path=/mnt/dfs/ssiingh/ACE/models/ACE_contrast_12n_global_inequal_swinb.pth
num_classes=14
batch_size=32
num_epochs=20
ckp_dir=/mnt/dfs/ssiingh/ACE/models/downstream_cls/ACE_pretrained_swinv1
result_file=/mnt/dfs/ssiingh/ACE/downstream/pretrained_swinv1_finetune_nih.txt
python downstream_cls.py --images_path $images_path --train_file_path $train_file_path --val_file_path $val_file_path --test_file_path $test_file_path --pretrained_model_path $pretrained_model_path --num_classes $num_classes --batch_size $batch_size --num_epochs $num_epochs --result_file $result_file --ckp_dir $ckp_dir