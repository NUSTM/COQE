# first stage.
CUDA_VISIBLE_DEVICES=5,6 python main.py --file_type="Car-COQE" --model_mode="bert" --program_mode="run" --stage_model="first" --epoch=25 --model_type="multitask" --embed_dropout=0.1 --premodel_path="/home/pretrain_model/"
CUDA_VISIBLE_DEVICES=5,6 python main.py --file_type="Ele-COQE" --model_mode="bert" --program_mode="run" --stage_model="first" --epoch=25 --model_type="multitask" --embed_dropout=0.1 --premodel_path="/home/pretrain_model/"
CUDA_VISIBLE_DEVICES=5,6 python main.py --file_type="Camera-COQE" --model_mode="bert" --program_mode="run" --stage_model="first" --epoch=25 --model_type="multitask" --embed_dropout=0.1 --premodel_path="/home/pretrain_model/"

# second and thrid stage.
CUDA_VISIBLE_DEVICES=5,6 python main.py --file_type="Car-COQE" --model_mode="bert" --program_mode="run" --stage_model="second" --epoch=25 --model_type="multitask" --embed_dropout=0.1 --premodel_path="/home/pretrain_model/" --factor=0.3
CUDA_VISIBLE_DEVICES=5,6 python main.py --file_type="Ele-COQE" --model_mode="bert" --program_mode="run" --stage_model="second" --epoch=25 --model_type="multitask" --embed_dropout=0.1 --premodel_path="/home/pretrain_model/" --factor=0.3
CUDA_VISIBLE_DEVICES=5,6 python main.py --file_type="Camera-COQE" --model_mode="bert" --program_mode="run" --stage_model="second" --epoch=25 --model_type="multitask" --embed_dropout=0.1 --premodel_path="/home/pretrain_model/" --factor=0.3
