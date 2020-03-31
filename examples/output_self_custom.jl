#=
Try to get a CNN to output the same image that went in
=#

using BSON: @save
using Flux
using Images
using Plots

include("../NeuralStyle/src/utils.jl")


@info "Loading images..."
img = load("../NeuralStyle/style_images/katsushika-hokusai_the-great-wave-off-kanagawa.jpg")
imgview = Float32.(channelview(img));  # Create a view of the image and convert
imgWHC = permutedims(imgview, (3, 2, 1)); # Images CHW order to Flux WHC(N) order


image_in = Array{Float32}(undef, (size(imgWHC)[1:2]..., 3, 1))  # Preallocate tensor
image_in[:, :, :, 1] = imgWHC  # Put RGB values into array

k = (3, 3);  # kernel size
s = (2, 2);  # stride size
d = (1, 1);  # dilation size

# Size and padding for the first layer
size1 = size(image_in)[1:2];
epad1 = pad2d(size1, k, s);
# Size and padding for the second layer
size2 = nstrides2d(size1, k, epad1, s);
epad2 = pad2d(size2, k, s);
# Size and padding for the third layer
size3 = nstrides2d(size2, k, epad2, s);
epad3 = pad2d(size3, k, s);
# Size and padding for the fourth layer
size4 = nstrides2d(size3, k, epad3, s);
epad4 = pad2d(size4, k, s);

autoencode = Chain(# Encoder section
               Conv(k, 3=>32, pad=epad1, stride=s, dilation=d),
               Conv(k, 32=>64, pad=epad2, stride=s, dilation=d),
               Conv(k, 64=>128, pad=epad3, stride=s, dilation=d),
               BatchNorm(128),
               Conv(k, 128=>256, relu, pad=epad4, stride=s, dilation=d),
               # Decoder section
               ConvTranspose(k, 256=>128, relu, pad=epad4, stride=s, dilation=d),
               BatchNorm(128),
               ConvTranspose(k, 128=>64, pad=epad3, stride=s, dilation=d),
               ConvTranspose(k, 64=>32, pad=epad2, stride=s, dilation=d),
               ConvTranspose(k, 32=>3, pad=epad1, stride=s, dilation=d)
               );


# Begin Training cyles
opt = Flux.ADAM()  # Standard optimizer

autoencode_loss(x) = Flux.mse(autoencode(x), x)

@info "Beginning training...."
losses = []
nepochs = 1000
for i=1:nepochs
    println("Training loop $i of $nepochs")
    Flux.train!(autoencode_loss, params(autoencode), [image_in], opt)
    push!(losses, autoencode_loss(image_in))
    println("Total loss = $(losses[end])")
end

@save "great_wave.bson" autoencode

@info "Plotting loss..."
p = plot(1:length(losses), losses)
display(p)

@info "Creating output image..."
# Resize the output model back to regular a regular image
im_autoencoded = autoencode(image_in)  # Get the final image dimension
outsize = size(im_autoencoded)[1:2]  # Remove the 4th dimension
output_features = reshape(im_autoencoded, (outsize..., 3))
output_perm = permutedims(output_features, (3, 2, 1)) # Flux WHC order to Images CHW orderr
image_out = colorview(RGB, output_perm) # Converts to an RGB image
