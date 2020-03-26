EXPERIMENT=focalstack

train:
	python deepdepth.py --experiment_name=$(EXPERIMENT) --gpu=1

baseline:
	python deepdepth.py --experiment_name=baseline --gpu=0

tensorboard:
	nohup tensorboard --bind_all --logdir ./experiments/$(EXPERIMENT) --samples_per_plugin images=10000 &

