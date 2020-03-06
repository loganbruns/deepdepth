

### Environment preparation steps

```
$ conda create --name deepdepth --file environment.yml
$ conda activate deepdepth
```

### Data preparation steps

#### Download NYUv2 depth labeled dataset (matlab data file)
```
$ cd data
$ wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
```

#### Convert Matlab data file into h5 format
Run the mat2h5.m script which will generate nyu\_depth\_v2.h5 file in the data directory.

#### Convert h5 format into tfrecord format

```
$ python h52tfrecord.py
```

This will generate the nyu\_depth\_v2.tfrecord file in the data directory.
