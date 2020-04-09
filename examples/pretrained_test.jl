using BSON: @load
using Flux
using Images
using NeuralStyle
using Plots

@info "Loading images..."
img = load("style_images/katsushika-hokusai_the-great-wave-off-kanagawa_1829.jpg");
imgview = Float32.(channelview(img));  # Create a view of the image and convert
imgWHC = permutedims(imgview, (3, 2, 1)); # Images CHW order to Flux WHC(N) order


image_in = Array{Float32}(undef, (size(imgWHC)[1:2]..., 3, 1));  # Preallocate tensor
image_in[:, :, :, 1] = imgWHC;  # Put RGB values into array

@load "pretrained_models/great_wave.bson" autoencode;

@info "Creating output image..."
# Resize the output model back to regular a regular image
im_autoencoded = autoencode(image_in);  # Get the final image dimension
outsize = size(im_autoencoded)[1:2];  # Remove the 4th dimension
output_features = reshape(im_autoencoded, (outsize..., 3));
output_perm = permutedims(output_features, (3, 2, 1)); # Flux WHC order to Images CHW orderr
image_out = colorview(RGB, output_perm) # Converts to an RGB image

@info "Creating layer image..."
# Resize the output model back to regular a regular image
content_img = load("content_images/Neckarfront_Tübingen_Mai_2017.jpg")
contentview = Float32.(channelview(content_img));  # Create a view of the image and convert
contentWHC = permutedims(contentview, (3, 2, 1)); # Images CHW order to Flux WHC(N) order

content_in = Array{Float32}(undef, (size(contentWHC)[1:3]..., 1));  # Preallocate tensor
content_in[:, :, :, 1] = contentWHC;  # Put RGB values into array


layerimage = autoencode(content_in);  # Get the final image dimension
outsize = size(layerimage)[1:2];  # Remove the 4th dimension
output_features = reshape(layerimage, (outsize..., 3));
output_perm = permutedims(output_features, (3, 2, 1)); # Flux WHC order to Images CHW orderr
image_out = colorview(RGB, output_perm) # Converts to an RGB image
