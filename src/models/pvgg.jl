# A pseudo-VGG19 model
# This model skips the max-pool layers and uses 2x2 strides to control image size
# Classification layers are left out.
# 16-layers (8 down, 8 up)
function pVGG_layers(image_size::Tuple{Int, Int})
    nlayers = 16
    kk = repeat([(3, 3)], nlayers);  # kernel size, all equal
    ss = repeat([(1, 1)], nlayers);  # stride size
    ss[2] = ss[4] = ss[8] = ss[12] = ss[16] = (2, 2)
    d = (1, 1);  # dilation size


    pc = padchain(nlayers, image_size, kk, ss)

    ae = Chain(
               # Encoder section
               # Block 1
               Conv(kk[1], 3=>64, relu, pad=pc[1], stride=ss[1], dilation=d),
               Conv(kk[2], 64=>64, relu, pad=pc[2], stride=ss[2], dilation=d),
               # Block 2
               Conv(kk[3], 64=>128, relu, pad=pc[3], stride=ss[3], dilation=d),
               Conv(kk[4], 128=>128, relu, pad=pc[4], stride=ss[4], dilation=d),
               # Block 3
               Conv(kk[5], 128=>256, relu, pad=pc[5], stride=ss[5], dilation=d),
               Conv(kk[6], 256=>256, relu, pad=pc[6], stride=ss[6], dilation=d),
               Conv(kk[7], 256=>256, relu, pad=pc[7], stride=ss[7], dilation=d),
               Conv(kk[8], 256=>256, relu, pad=pc[8], stride=ss[8], dilation=d),
               # Block 4
               Conv(kk[9], 256=>512, relu, pad=pc[9], stride=ss[9], dilation=d),
               Conv(kk[10], 512=>512, relu, pad=pc[10], stride=ss[10], dilation=d),
               Conv(kk[11], 512=>512, relu, pad=pc[11], stride=ss[11], dilation=d),
               Conv(kk[12], 512=>512, relu, pad=pc[12], stride=ss[12], dilation=d),
               # Block 5
               Conv(kk[13], 512=>512, relu, pad=pc[13], stride=ss[13], dilation=d),
               Conv(kk[14], 512=>512, relu, pad=pc[14], stride=ss[14], dilation=d),
               Conv(kk[15], 512=>512, relu, pad=pc[15], stride=ss[15], dilation=d),
               Conv(kk[16], 512=>512, relu, pad=pc[16], stride=ss[16], dilation=d),
               # Decoder section
               # Block 5
               ConvTranspose(kk[16], 512=>512, relu, pad=pc[16], stride=ss[16], dilation=d),
               ConvTranspose(kk[15], 512=>512, relu, pad=pc[15], stride=ss[15], dilation=d),
               ConvTranspose(kk[14], 512=>512, relu, pad=pc[14], stride=ss[14], dilation=d),
               ConvTranspose(kk[13], 512=>512, relu, pad=pc[13], stride=ss[13], dilation=d),
               # Block 4
               ConvTranspose(kk[12], 512=>512, relu, pad=pc[12], stride=ss[12], dilation=d),
               ConvTranspose(kk[11], 512=>512, relu, pad=pc[11], stride=ss[11], dilation=d),
               ConvTranspose(kk[10], 512=>512, relu, pad=pc[10], stride=ss[10], dilation=d),
               ConvTranspose(kk[9], 512=>256, relu, pad=pc[9], stride=ss[9], dilation=d),
               # Block 3
               ConvTranspose(kk[8], 256=>256, relu, pad=pc[8], stride=ss[8], dilation=d),
               ConvTranspose(kk[7], 256=>256, relu, pad=pc[7], stride=ss[7], dilation=d),
               ConvTranspose(kk[6], 256=>256, relu, pad=pc[6], stride=ss[6], dilation=d),
               ConvTranspose(kk[5], 256=>128, relu, pad=pc[5], stride=ss[5], dilation=d),
               # Block 2
               ConvTranspose(kk[4], 128=>128, relu, pad=pc[4], stride=ss[4], dilation=d),
               ConvTranspose(kk[3], 128=>64, relu, pad=pc[3], stride=ss[3], dilation=d),
               # Block 1
               ConvTranspose(kk[2], 64=>64, relu, pad=pc[2], stride=ss[2], dilation=d),
               ConvTranspose(kk[1], 64=>3, relu, pad=pc[1], stride=ss[1], dilation=d)
               );

    return ae
end

struct pVGG <: AutoEncoderModel
    layers::Chain
end

pVGG(image_size::Tuple{Int, Int}) = pVGG(pVGG_layers(image_size))

Base.show(io::IO, ::pVGG) = print(io, "pVGG()")

Flux.@functor pVGG

(m::pVGG)(x) = m.layers(x)
