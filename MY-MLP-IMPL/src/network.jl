relu(x) = max.(0, x)

struct Dense
    W::Matrix{Float32}
    b::Array{Float32,1}
    σ::Function
end

function Dense(in::Int, out::Int, σ::Function)
    W = randn(Float32, out, in)
    b = zeros(Float32, out)
    Dense(W, b, σ)
end

function (layer::Dense)(x)
    layer.σ(layer.W * x .+ layer.b)
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
