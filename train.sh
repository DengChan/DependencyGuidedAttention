python train.py --id 00 --cache_dir ./pretrained --model_name_or_path bert-base-cased --batch_size 32 --lr 0.000005 --match_loss_weight 0.0 --fintune_bert True
python train.py --id 01 --cache_dir ./pretrained --model_name_or_path bert-base-cased --batch_size 32 --lr 0.000001 --match_loss_weight 0.0 --fintune_bert True
python train.py --id 02 --cache_dir ./pretrained --model_name_or_path bert-base-cased --batch_size 32 --lr 0.00001 --match_loss_weight 0.0 --fintune_bert True
python train.py --id 03 --cache_dir ./pretrained --model_name_or_path bert-base-cased --batch_size 32 --lr 0.000005 --match_loss_weight 1.0 --fintune_bert True
