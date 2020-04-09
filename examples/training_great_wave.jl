#=
Try to get a CNN to output the same image that went in
=#

using BSON: @save
using Dates
using Flux
using Images
using NeuralStyle
using Plots

@info "Loading images..."
img = load("style_images/katsushika-hokusai_the-great-wave-off-kanagawa_1829.jpg")
imgview = Float32.(channelview(img));  # Create a view of the image and convert
imgWHC = permuteddimsview(imgview, (3, 2, 1)); # Images CHW order to Flux WHC(N) order

image_in = Array{Float32}(undef, (size(imgWHC)[1:2]..., 3, 1))  # Preallocate tensor
image_in[:, :, :, 1] = imgWHC  # Put RGB values into array

@info "Setting up model..."
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
# Size and padding for the fifth layer
size5 = nstrides2d(size4, k, epad4, s);
epad5 = pad2d(size5, k, s);

autoencode = Chain(# Encoder section
               Conv(k, 3=>32, relu, pad=epad1, stride=s, dilation=d),
               Conv(k, 32=>64, relu, pad=epad2, stride=s, dilation=d),
               Conv(k, 64=>128, relu, pad=epad3, stride=s, dilation=d),
               Conv(k, 128=>256, relu, pad=epad4, stride=s, dilation=d),
               Conv(k, 256=>512, relu, pad=epad5, stride=s, dilation=d),
               # Decoder section
               ConvTranspose(k, 512=>256, relu, pad=epad5, stride=s, dilation=d),
               ConvTranspose(k, 256=>128, relu, pad=epad4, stride=s, dilation=d),
               ConvTranspose(k, 128=>64, relu, pad=epad3, stride=s, dilation=d),
               ConvTranspose(k, 64=>32, relu, pad=epad2, stride=s, dilation=d),
               ConvTranspose(k, 32=>3, relu, pad=epad1, stride=s, dilation=d)
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


@info "Saving model..."
rightnow = now()  # Do this to have all times be equal
@save "pretrained_models/great_wave_model-$rightnow.bson" autoencode
@save "pretrained_models/great_wave_loss-$rightnow.bson" losses

@info "Plotting loss..."
p = plot(title="Great Wave of Kanagawa",
         xaxis = ("Training Cycles"),
         xlim = (1, 1000),
         yaxis = ("Training Loss", 1:length(losses), :log10),
         ylim = (1e-3, 1e0),
         yticks = (1e-3, 1e-1, 1e0))
p = plot!(1:length(losses), losses, label="Training Loss")
display(p)
savefig(p, "examples/great_wave_loss-$rightnow.png")

@info "Creating output image..."
# Resize the output model back to regular a regular image
im_autoencoded = autoencode(image_in)
outsize = size(im_autoencoded)[1:2]  # Get the final image dimension
output_features = reshape(im_autoencoded, (outsize..., 3))  # Remove the 4th dimension
output_perm = permuteddimsview(output_features, (3, 2, 1)) # Flux WHC order to Images CHW orderr
image_out = colorview(RGB, output_perm) # Converts to an RGB image

save("examples/great_wave_CNN-$rightnow.png", image_out)
