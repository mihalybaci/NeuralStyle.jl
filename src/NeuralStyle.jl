module NeuralStyle

using BSON: @load, @save
using Dates
using Flux
using Images
using Plots
using Random

# Exports from BSON
export @load, @save

# Basic utilities
include("utils.jl")
export nstrides1d, nstrides2d, pad1d, pad2d, tensor2image, image2tensor

# Provides locations for included contents and styles
include("datasets.jl")
export list_contents, list_styles

# Homegrown models
include("models/abstractmodel.jl")
export AutoEncoderModel
include("models/aem5.jl")
export AEM5
include("models/pvgg.jl")
export pVGG


end # module
