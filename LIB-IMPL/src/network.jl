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


struct MaxPool{N,M}
  k::NTuple{N,Int}
  pad::NTuple{M,Int}
  stride::NTuple{N,Int}
end

function MaxPool(k::NTuple{N,Integer}; pad = 0, stride = k) where N
  stride = expand(Val(N), stride)
  pad = calc_padding(MaxPool, pad, k, 1, stride)
  return MaxPool(k, pad, stride)
end

function (m::MaxPool)(x)
  return maxpool(x, m.k, m.pad, m.stride)
end

struct SamePad end

calc_padding(lt, pad, k::NTuple{N,T}, dilation, stride) where {T,N} = expand(Val(2*N), pad)
function calc_padding(lt, ::SamePad, k::NTuple{N,T}, dilation, stride) where {N,T}
  k_eff = @. k + (k - 1) * (dilation - 1)
  pad_amt = @. k_eff - 1
  return Tuple(mapfoldl(i -> [cld(i, 2), fld(i,2)], vcat, pad_amt))
end

struct PoolDims{N, K, S, P, D}
    input_size::NTuple{N, Int}

    kernel_size::NTuple{K, Int}
    channels_in::Int

    stride::NTuple{S, Int}
    padding::NTuple{P, Int}
    dilation::NTuple{D, Int}
end

function PoolDims(
    x_size::NTuple{M}, k::Union{NTuple{L, Int}, Int};
    stride = k, padding = 0, dilation = 1,
) where {M, L}
    _check_kernel(k::Number, N::Int) = ntuple(_ -> Int(k), N)
    _check_kernel(k::NTuple, ::Int) = k

    kernel = _check_kernel(k, M - 2)
    length(x_size) == length(kernel) + 2 || error(
        "PoolDims expects ndim(x) == length(k)+2 or length(size(x)) == length(kernel)+2,
        dimension of x_size is $(length(x_size)),
        length of k need $(length(x_size) - 2),
        but now it's $(length(kernel))"
    )
    spdf_kernel = NTuple{M, Int}([kernel..., 1, 1])

    sstride, ppadding, ddilation = check_spdf(
        x_size, spdf_kernel, stride, padding, dilation)
    PoolDims(
        x_size[1:(end - 2)], kernel, x_size[end - 1],
        sstride, ppadding, ddilation)
end

PoolDims(x::AbstractArray, k; kwargs...) = PoolDims(size(x), k; kwargs...)