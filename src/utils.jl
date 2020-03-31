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
l - length
k - kernel size
s - stride

Output:
----
A 2-element tuple with the number of elements needed to pad each side of a 1-D length.
"""
function pad1d(l, k, s)
    n = ceil((l - k)/s + 1)  # Number of whole kernels needed to cross l
    pad_tot = s*(n - 1) + k - l  # Padding needed to get to n kernels
    # If padding still does not give an inter number of kernel strides
    # Then print an error
    if ((l - k + pad_tot) / s + 1) % 1 != 0
        @error "Could not pad array properly"
    end
    # Try to evenly space padding between sides
    # left, right, top, and bottom sides.
    # The following pads bottom/right sides first
    p1 = pad_tot ÷ 2
    p2 = pad_tot - p1

    return convert.(Int, (p1, p2))
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
