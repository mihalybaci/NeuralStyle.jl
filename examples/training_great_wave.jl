#=
Try to get a CNN to output the same image that went in
=#

#using BSON: @save
#using Dates
#using Flux
#using Images
using NeuralStyle
#using Plots

@info "Loading images..."
image_in = load("style_images/katsushika-hokusai_the-great-wave-off-kanagawa_1829.jpg")
tensor_in = image2tensor(image_in);


ten_dim = size(tensor_in)[1:2]  # Model requires tensor width/height to determine padding
autoenocde = AutoCNN(ten_dim)  # Load autoencoder model

# Begin Training cyles
opt = Flux.ADAM()  # Standard optimizer

autoencode_loss(x) = Flux.mse(autoencode(x), x)

@info "Beginning training...."
losses = []
nepochs = 1
for i=1:nepochs
    println("Training loop $i of $nepochs")
    Flux.train!(autoencode_loss, params(autoencode), [tensor_in], opt)
    push!(losses, autoencode_loss(tensor_in))
    println("Total loss = $(losses[end])")
end

@info "Saving model..."
rightnow = Dates.now()  # Do this to have all times be equal
@save "pretrained_models/great_wave_model-$nepochs-$rightnow.bson" autoencode
@save "pretrained_models/great_wave_loss-$nepochs-$rightnow.bson" losses

@info "Plotting loss..."
p = plot(title="Great Wave of Kanagawa",
         xaxis = ("Training Cycles"),
         xlim = (1, 1000),
         yaxis = ("Training Loss", 1:length(losses), :log10),
         ylim = (1e-3, 1e0),
         yticks = (1e-3, 1e-1, 1e0))
p = plot!(1:length(losses), losses, label="Training Loss")
display(p)
savefig(p, "examples/great_wave_loss-$nepochs-$rightnow.png")

@info "Creating output image..."
tensor_out = autoencode(tensor_in)  # Run the autoencoder on the input image
image_out = tensor2image(tensor_out)  # Create the output image

save("examples/great_wave_CNN-$nepochs-$rightnow.png", image_out)
