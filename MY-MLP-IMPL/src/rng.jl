using Random

function glorot_uniform(rng::AbstractRNG, dims::Integer...; gain::Real=1)
    scale = Float32(gain) * sqrt(24.0f0 / sum(nfan(dims...)))
    (rand(rng, Float32, dims...) .- 0.5f0) .* scale
end
glorot_uniform(dims::Integer...; kw...) = glorot_uniform(Random.default_rng(), dims...; kw...)
glorot_uniform(rng::AbstractRNG=Random.default_rng(); init_kwargs...) = (dims...; kwargs...) -> glorot_uniform(rng, dims...; init_kwargs..., kwargs...)

nfan() = 1, 1 # fan_in, fan_out
nfan(n) = 1, n # A vector is treated as a n×1 matrix
nfan(n_out, n_in) = n_in, n_out # In case of Dense kernels: arranged as matrices
nfan(dims::Tuple) = nfan(dims...)
nfan(dims...) = prod(dims[1:end-2]) .* (dims[end-1], dims[end]) # In case of convolution kernels

