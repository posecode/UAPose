# UAPose
## Uncertainty-Aware Representation Learning for 2D Human Pose Estimation in Videos

### Environment and datasets
For experimental environment and dataset preparation, please refer to https://github.com/Pose-Group/DCPose/blob/main/docs/Installation.md

### Training from scratch
#### For PoseTrack2017
```ruby
require 'redcarpet'
markdown = Redcarpet.new("Hello World!")
puts markdown.to_html
```

cd tools
# train  
python run.py --cfg ../configs/posetimation/DcPose/posetrack17/no_supp_targ_vit_large.yaml --train

# val 
python run.py --cfg ../configs/posetimation/DcPose/posetrack17/no_supp_targ_vit_large.yaml --val 

