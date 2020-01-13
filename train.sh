# ¶Ô±Èloss Ó°Ïì
# python train.py --id 20 --cache_dir ./pretrained --model_name_or_path bert-base-cased --dga_layers 2 --lr 0.000005 --match_loss_weight 0.0 --fintune_bert True --num_epoch 10

# python train.py --id 23 --cache_dir ./pretrained --load True --model_file ./saved_models/22/best_model.pt --model_name_or_path bert-base-cased --dga_layers 1 --lr 0.000005 --neg_weight 1.0 --fintune_bert True --num_epoch 10

# python train.py --id 24 --cache_dir ./pretrained --model_name_or_path bert-base-cased --dga_layers 1 --lr 0.000005 --neg_weight 1.0 --fintune_bert True --num_epoch 10

python train.py --id 33 --cache_dir ./pretrained --model_name_or_path bert-base-cased --K_dim 32 --dga_layers 4 --lr 0.000005 --neg_weight 1.0 --fintune_bert True --num_epoch 10

# python train.py --id 31 --cache_dir ./pretrained --model_name_or_path bert-base-cased --dga_layers 3 --lr 0.000005 --neg_weight 1.0 --fintune_bert True --num_epoch 10