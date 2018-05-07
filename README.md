# HDR
This project is used for deghosting artifacts in HDR imging method.

At first, we select a reference image with most well-exposed pixels. Then we 
use both color mapping and patchmatch to generate some latent imagess. In the
end, the latent images and the reference image are used for HDR imaging. The
main idea is credited to Jun Hu, et al. "HDR deghosting: How to deal with 
saturation?" CVPR(2013). We simplified some prodcedures and also changed some
steps.

Please read General.py to see technique details.
