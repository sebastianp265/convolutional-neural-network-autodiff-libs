function gather(W::AbstractMatrix, idxs::AbstractVector{<:Integer})
    emb_dim = size(W, 1)
    out = Matrix{eltype(W)}(undef, emb_dim, length(idxs))
    for (j, idx) in enumerate(idxs)
        out[:, j] = W[:, idx]
    end
    return out
end

to_3d(x::AbstractArray, s::NTuple{2,Integer}) = reshape(x, :, s[1], s[2]) 