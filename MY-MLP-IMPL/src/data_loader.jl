using Random

mutable struct DataLoader
    data::Tuple
    batchsize::Int
    shuffle::Bool

    function DataLoader(data::Tuple; batchsize::Int, shuffle::Bool=true)
        if shuffle
            data = tuple([g[:, randperm(size(g)[2])] for g in data]...)
        end
        new(data, batchsize, shuffle)
    end
end

function Base.iterate(dl::DataLoader, state=1)
    data = dl.data
    batch_size = dl.batchsize

    n_samples = size(data[1])[2]
    n_batches = ceil(Int, n_samples / batch_size)

    if state > n_batches
        return nothing
    end

    start_idx = (state - 1) * batch_size + 1
    end_idx = min(state * batch_size, n_samples)

    batch = tuple([g[:, start_idx:end_idx] for g in data]...)
    return batch, state + 1
end
