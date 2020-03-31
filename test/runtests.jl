using NeuralStyle
using Test

@info "Testing NeuralStyle.jl ..."

@testset "NeuralStyle" begin

  @testset "Utils" begin
    include("utils.jl")
  end

end
