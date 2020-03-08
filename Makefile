EXPERIMENT=baseline

train:
	CUDA_VISIBLE_DEVICES=0 python deepdepth.py

tensorboard:
	nohup tensorboard --bind_all --logdir ./experiments/$(EXPERIMENT)/logs &

