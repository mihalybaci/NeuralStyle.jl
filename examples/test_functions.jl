#=
Make simple functions to:
 - Turn images into Flux tensors
 - make padding easier if the number of convolutions in a model changes
=#

using Flux
using NeuralStyle

@info "Loading images..."
image_in = load("style_images/katsushika-hokusai_the-great-wave-off-kanagawa_1829.jpg")
tensor_in = image2tensor(image_in);

ten_dim = size(tensor_in)[1:2]  # Model requires tensor width/height to determine padding
auto = AutoCNN(ten_dim)

# Begin Training cyles
opt = Flux.ADAM()  # Standard optimizer

autoencode_loss(x) = Flux.mse(auto(x), x)

@info "Beginning training...."
losses = []
nepochs = 1
for i=1:nepochs
    println("Training loop $i of $nepochs")
    Flux.train!(autoencode_loss, params(auto), [tensor_in], opt)
    push!(losses, autoencode_loss(tensor_in))
    println("Total loss = $(losses[end])")
end
