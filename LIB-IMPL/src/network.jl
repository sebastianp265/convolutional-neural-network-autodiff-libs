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
    weight::W
end

Embedding(in::Integer, out::Integer; init=randn32) = Embedding(init(out, in))

(m::Embedding)(x::AbstractArray) = reshape(gather(m.weight, vec(x)), :, size(x)...)

function gather(W::AbstractMatrix, idxs::AbstractVector{<:Integer})
    emb_dim = size(W, 1)
    out = Matrix{eltype(W)}(undef, emb_dim, length(idxs))
    for (j, idx) in enumerate(idxs)
        out[:, j] = W[:, idx]
    end
    return out
end

struct Chain
    layers::Tuple
end

function Chain(layers...)
    Chain(layers)
end

function (c::Chain)(x)
    for layer in c.layers
        x = layer(x)
    end
    return x
end
