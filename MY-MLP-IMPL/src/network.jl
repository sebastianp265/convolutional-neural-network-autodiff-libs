struct Dense{T}
    weight::Variable{Matrix{T}}
    bias::Variable{Vector{T}}
    σ::Function
end

function Dense(in::Int, out::Int, σ::Function)
    W = Variable(randn(Float32, out, in), "W($in,$out)")
    b = Variable(zeros(Float32, out), "b($in,$out)")
    Dense(W, b, σ)
end

function (layer::Dense)(x)
    layer.σ.(layer.weight * x .+ layer.bias)
end

struct Chain
    layers::Tuple{Vararg{Dense}}
end

function Chain(layers::Dense...)
    Chain(layers)
end

function (c::Chain)(x)
    for layer in c.layers
        x = layer(x)
    end
    return x
end
