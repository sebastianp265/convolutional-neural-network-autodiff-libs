struct Dense{F,W<:AbstractMatrix,B}
    weight::Variable{W}
    bias::Variable{B}
    σ::F
    function Dense(in::Integer, out::Integer, σ::F; init=glorot_uniform) where {F}
        W = init(out, in)
        b = create_bias(W, size(W, 1))
        new{F,typeof(W),typeof(b)}(Variable(W), Variable(b), σ)
    end
end

function create_bias(W::AbstractArray, dims::Integer...)
    fill!(similar(W, dims...), 0)
end

function (layer::Dense)(x)
    layer.σ.(layer.weight * x .+ layer.bias)
end

struct Embedding{W<:AbstractMatrix}
    weight::Variable{W}
end

Embedding(in::Integer, out::Integer; init=randn32) = Embedding(Variable(init(out, in)))

function (m::Embedding)(x::AbstractArray)
    gathered = gather(m.weight, vec(x))
    to_3d(gathered, size(x))
end

struct Chain
    layers::Tuple
    Chain(layers...) = new(layers)
end

function (c::Chain)(x)
    for layer in c.layers
        x = layer(x)
    end
    return x
end

struct Conv{N,M,F,W<:AbstractArray,B}
    σ::F
    weight::Variable{W}
    bias::Variable{B}
    stride::NTuple{N,Int}
    pad::NTuple{M,Int}
    dilation::NTuple{N,Int}
    groups::Int
end

function Conv(k::NTuple{1,Integer}, ch::Pair{<:Integer,<:Integer}, σ=identity; 
             init=glorot_uniform, stride=1, pad=0, dilation=1, groups=1)
    in, out = ch.first, ch.second
    weight = init(k..., in÷groups, out)
    bias = create_bias(weight, out)
    # Convert scalar parameters to 1-tuples for 1D convolution
    stride_tuple = Tuple(expand(1, stride))
    pad_tuple = Tuple(expand(2, pad))  # 2-tuple for 1D conv (before and after)
    dilation_tuple = Tuple(expand(1, dilation))
    Conv{1,2,typeof(σ),typeof(weight),typeof(bias)}(σ, Variable(weight), Variable(bias), 
        stride_tuple, pad_tuple, dilation_tuple, groups)
end

function (c::Conv)(x)
    conv1d(c.weight, x, c.bias, c.σ, c.stride, c.pad, c.dilation, c.groups)
end
