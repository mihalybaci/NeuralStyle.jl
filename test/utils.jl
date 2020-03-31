using Flux
using NeuralStyle
using Random
using Test

@testset "Image Size" begin
# Make a set of randomly sized arrays with random kernels/strides
# and ensure that the input and output sizes are equal after convolutions
    @info "Testing image size preservation"
    equal_size = []  # Array to hold boolean values
    for i in 1:10
        # Make an array of random floats with random sizes
        input_tensor =  rand(rand(100:500), rand(100:500), 3, 1);
        size_in = size(input_tensor)[1:2]  # size of input array

        k = (rand(1:9), rand(1:9))  # kernel size
        s = (rand(1:9), rand(1:9))  # stride size
        d = (1, 1)
        epad1 = pad2d(size_in, k, s)  # Basic padding calculation
        layer2size = nstrides2d(size_in, k, epad1, s)  # Estimate the size of layer 2
        epad2 = pad2d(layer2size, k, s)

        autoencode = Chain(Conv(k, 3=>8, relu, pad=epad1, stride=s, dilation=d),
                           Conv(k, 8=>16, relu, pad=epad2, stride=s, dilation=d),
                           ConvTranspose(k, 16=>8,relu, pad=epad2, stride=s, dilation=d),
                           ConvTranspose(k, 8=>3, relu, pad=epad1, stride=s, dilation=d))

        #println("\nPrinting results")
        encoded = autoencode(input_tensor)
        push!(equal_size, size(input_tensor) == size(encoded))
    end
    @test sum(equal_size) == 10  # Should be 10, (true=1)
end
