# Why we didn't use the A4 segmentation model
- The segmentation model was not used because it was not able to accurately segment video frames.
- It was trained on very limited data primary only in very urban city areas.
- Another issue was the segmentation model just did not know how to deal with dashboard glares.
- Even if we just gave the model the road, it would still not be able to accurately segment the road.
- The model was trained on images and not on video frames, which is why it was not able to accurately segment video frames.
- Ideally if we used ML for this problem, we would want a model that also takes into account the temporal information of the video frames.
