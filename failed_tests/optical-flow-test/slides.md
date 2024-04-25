Why optical flow didn't work:
- Aperture problem: the road color is uniform, so the optical flow algorithm can't detect the motion. This is because the motion of the road is perpendicular to the gradient of the image.
- Optical flow is also somewhat slow unless we use a space feature set