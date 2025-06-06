# UAPose
## Uncertainty-Aware Representation Learning for 2D Human Pose Estimation in Videos

### Environment and datasets
For experimental environment and dataset preparation, please refer to https://github.com/Pose-Group/DCPose/blob/main/docs/Installation.md

### Training from scratch
#### For PoseTrack2017
```ruby
cd tools
# train  
python run.py --cfg ../configs/posetimation/DcPose/posetrack17/no_supp_targ_vit_large.yaml --train
```
### Validation
```ruby
cd tools
# val  
python run.py --cfg ../configs/posetimation/DcPose/posetrack17/no_supp_targ_vit_large.yaml --val
```
### Acknowledgement
Our code refers to the following repositories:

- [DCPose](https://github.com/Pose-Group/DCPose)
- [FAMI-Pose](https://github.com/Pose-Group/FAMI-Pose)


We thank the authors for releasing their codes.
