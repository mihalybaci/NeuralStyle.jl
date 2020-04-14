"""
nstrides1d(l, k, p, s)

nstrides1d is a function to calculate the number of kernel strides that fit in a given
length. This equals the size of a dimension after a convolution.
"""
nstrides1d(l, k, p, s) = ((l - k + p) / s + 1)


"""
nstrides2d(l, k, p, s)

nstrides2d is a convenience function that accepts tuple inputs in the manner expected for
the pad1d and pad2d functions and outpus 2-element tuple.
"""
nstrides2d(l, k, p, s) = (nstrides1d(l[1], k[1], p[1] + p[2], s[1]),
                          nstrides1d(l[2], k[2], p[3] + p[4], s[2]))


"""
pad1d(l, k, s)

Calculate the amount of padding needed to fit an integer number of kernel strides
across a given length. While pad_1d works for 2-D inputs, it is suggested to use
the helper function pad2d.

Inputs:
-------
l - length\n
k - kernel size\n
s - stride\n

Output:
----
A 2-element tuple with the number of elements needed to pad each side of a 1-D length.
"""
function pad1d(l, k, s)
    n = ceil((l - k)/s + 1)  # Number of whole kernels needed to cross l
    pad_tot = s*(n - 1) + k - l  # Padding needed to get to n kernels
    # If padding still does not give an inter number of kernel strides
    # Then print an error
    @info "$l  $k  $s"
    if ((l - k + pad_tot) / s + 1) % 1 != 0
        @error "Could not pad array properly"
    end
    # Try to evenly space padding between sides
    # left, right, top, and bottom sides.
    # The following pads bottom/right sides first
    @info "pad_tot = $pad_tot  |  div = $(pad_tot ÷ 2)"
    p1 = convert(Int, pad_tot ÷ 2)
    p2 = convert(Int, pad_tot) - p1

    return (p1, p2)
end

"""
pad2d(l, k, s)

Convience function for pad_1d. This function splats the output into a single,
N-element tuple rather than using pad_1d.(x, y, z), which outputs a tuple of tuples

Inputs:
-------
l - length
k - kernel size
s - stride

Output:
----
A 4-element tuple with the number of elements needed to pad each side of a 2-D matrix.
"""
pad2d(l, k, s) = (pad1d(l[1], k[1], s[1])...,  pad1d(l[2], k[2], s[2])...)

"""
padchain(N, l, k, s)

A function for outputting an array of padding tuples for use in Flux Chain models
"""
function padchain(N, l, k, s)
    ls = Array{Tuple}(undef, N)
    ps = Array{Tuple}(undef, N)

    ls[1] = l
    ps[1] = pad2d(ls[1], k[1], s[1])
    for i = 2:N
        ls[i] = nstrides2d(ls[i-1], k[i-1], ps[i-1], s[i-1])
        ps[i] = pad2d(ls[i], k[i], s[i])
    end

    return ps
end


"""
image2tensor(image)

Converts an input greyscale or RGB image into a tensor suitable for use by Flux.
"""
function image2tensor(image)
    # Using views is 2.5x faster and makes 2x fewer allocations
    imsize = size(image)
    # In one line, turn a an image type into array, convert to Float32, and rearrange
    array = permuteddimsview(Float32.(channelview(image)), (3, 2, 1))
    tensor = reshape(array, (size(array)..., 1))

    return tensor
end


"""
tensor2image(tensor)

Converts an input tensor suitable into greyscale or RGB image.
"""
function tensor2image(tensor)
    if size(tensor)[3] == 1  # Grayscale
        image = colorview(Gray, permuteddimsview(reshape(tensor, size(tensor)[1:3]), (3, 2, 1)))
    elseif size(tensor)[3] == 3
        image = colorview(RGB, permuteddimsview(reshape(tensor, size(tensor)[1:3]), (3, 2, 1)))
    else
        @error "Number of image channels ($(size(image)[3])) not recognized (1 = grayscale, 3 = RGB)"
    end

    return image
end
