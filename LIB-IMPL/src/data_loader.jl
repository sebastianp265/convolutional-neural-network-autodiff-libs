using Random

mutable struct DataLoader{D}
    data::D
    batchsize::Int
    shuffle::Bool
    rng::AbstractRNG
end

function DataLoader(
    data::Tuple;
    batchsize::Int,
    shuffle::Bool=true,
    rng::AbstractRNG=Random.default_rng()
)
    return DataLoader(data, batchsize, shuffle, rng)
end

struct DataLoaderIterator{D}
    loader::DataLoader{D}
    order::Vector{Int}
    position::Int
end

function Base.iterate(iter::DataLoaderIterator, state=iter.position)
    data = iter.loader.data
    batchsize = iter.loader.batchsize
    order = iter.order
    n_samples = length(order)

    if state > n_samples
        return nothing
    end

    end_idx = min(state + batchsize - 1, n_samples)
    batch_indices = order[state:end_idx]

    batch = tuple([g[:, batch_indices] for g in data]...)
    return batch, end_idx + 1
end

function Base.iterate(dl::DataLoader, state=1)
    order = dl.shuffle ? randperm(dl.rng, size(dl.data[1], 2)) : collect(1:size(dl.data[1], 2))
    iter = DataLoaderIterator(dl, order, state)
    return iterate(iter)
end

Base.IteratorSize(::Type{<:DataLoader}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:DataLoader}) = Base.EltypeUnknown()

