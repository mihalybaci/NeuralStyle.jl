# 10-layer (5 down, 5 up) autoencoder model (AEM)
function aem5_layers(image_size::Tuple{Int, Int})
    k = (3, 3);  # kernel size
    s = (2, 2);  # stride size
    d = (1, 1);  # dilation size

    # Size and padding for the first layer
    epad1 = pad2d(image_size, k, s);
    # Size and padding for the second layer
    size2 = nstrides2d(image_size, k, epad1, s);
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

    autoencode = Chain(
                   # Encoder section
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

    return autoencode
end

struct AEM5 <: AutoEncoderModel
    layers::Chain
end

AEM5(image_size::Tuple{Int, Int}) = AEM5(aem5_layers(image_size))

Base.show(io::IO, ::AEM5) = print(io, "AEM5()")

Flux.@functor AEM5

(m::AEM5)(x) = m.layers(x)
