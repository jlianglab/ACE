images_path=/data/jliang12/shared/dataset/NIHCXR14/full_images/images
train_file_path=/scratch/ssiingh/JLiangLab/BenchmarkArk/dataset/Xray14_train_official.txt
val_file_path=/scratch/ssiingh/JLiangLab/BenchmarkArk/dataset/Xray14_val_official.txt
test_file_path=/scratch/ssiingh/JLiangLab/BenchmarkArk/dataset/Xray14_test_official.txt
pretrained_model_path=/scratch/ssiingh/JLiangLab/ACE/models/ACE_contrast_12n_global_inequal_swinb.pth
num_classes=14
batch_size=3
num_epochs=20
result_file=/scratch/ssiingh/JLiangLab/ACE/downstream/pretrained_swinv1_finetune_nih.txt
python downstream_cls.py --images_path $images_path --train_file_path $train_file_path --val_file_path $val_file_path --test_file_path $test_file_path --pretrained_model_path $pretrained_model_path --num_classes $num_classes --batch_size $batch_size --num_epochs $num_epochs --result_file $result_file