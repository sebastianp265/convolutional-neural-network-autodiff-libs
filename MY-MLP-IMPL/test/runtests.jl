using Test
using MYMLP
using Statistics
import Flux
using TimerOutputs
import Base: ==

==(x::Flux.Optimisers.Leaf, y::MYMLP.Leaf) = x.rule == x.rule && x.state == y.state
==(x::Flux.Optimisers.Adam, y::MYMLP.Adam) = x.eta == y.eta && x.beta == y.beta && x.epsilon == y.epsilon
==(x::Flux.Chain, y::MYMLP.Chain) = all([a == b for (a, b) in zip(x.layers, y.layers)])
==(x::Flux.Dense, y::MYMLP.Dense) = x.weight == y.weight.output && x.bias == y.bias.output

==(x::Constant{X}, y::Constant{Y}) where {X,Y} =
    X === Y && x.output == y.output
==(x::Variable{X}, y::Variable{Y}) where {X,Y} =
    X === Y && x.output == y.output && x.gradient == y.gradient
==(x::Operator{FX,X}, y::Operator{FY,Y}) where {FX,FY,X,Y} =
    FX === FY && X === Y &&
    x.output == y.output &&
    x.gradient == y.gradient &&
    all([a == b for (a, b) in zip(x.inputs, y.inputs)])


is_base_test = false
#is_base_test = true
if is_base_test
    include("computional-graph.jl")
    include("gradient.jl")
    include("data_loader.jl")
    include("optimisers.jl")
    include("rng.jl")
else # acceptance test
    include("KM2.jl")
end

