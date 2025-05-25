expand(N, i::Tuple) = i
expand(N, i::Integer) = ntuple(_ -> i, N)

function gather(W::AbstractMatrix, idxs::AbstractVector{<:Integer})
    emb_dim = size(W, 1)
    out = Matrix{eltype(W)}(undef, emb_dim, length(idxs))
    for (j, idx) in enumerate(idxs)
        out[:, j] = W[:, idx]
    end
    return out
end

to_3d(x::AbstractArray, s::NTuple{2,Integer}) = reshape(x, :, s[1], s[2])

function conv1d(W::AbstractArray{T,3}, x::AbstractArray{T,3}, b::AbstractVector{T}, σ,
    stride::NTuple{1,Int}, pad::NTuple{2,Int}, dilation::NTuple{1,Int}, groups) where T
    kernel_size, in_channels_per_group, out_channels = size(W)
    seq_len, features, batch_size = size(x)

    @assert features == in_channels_per_group * groups "Input channels must match weight dimensions"

    if pad[1] > 0 || pad[2] > 0
        padded_x = zeros(T, seq_len + pad[1] + pad[2], features, batch_size)
        padded_x[pad[1]+1:pad[1]+seq_len, :, :] = x
        x = padded_x
        seq_len = size(x, 1)
    end

    out_seq_len = div(seq_len - (kernel_size - 1) * dilation[1] - 1, stride[1]) + 1
    result = zeros(T, out_seq_len, out_channels, batch_size)

    for batch in 1:batch_size
        for t in 1:out_seq_len
            t_start = (t - 1) * stride[1] + 1
            for out_ch in 1:out_channels
                for g in 1:groups
                    in_ch_start = (g - 1) * in_channels_per_group + 1
                    in_ch_end = g * in_channels_per_group
                    for in_ch_offset in 1:in_channels_per_group
                        in_ch = in_ch_start + in_ch_offset - 1
                        kernel_slice = @view W[:, in_ch_offset, out_ch]
                        for k in 1:kernel_size
                            t_pos = t_start + (k - 1) * dilation[1]
                            result[t, out_ch, batch] += kernel_slice[k] * x[t_pos, in_ch, batch]
                        end
                    end
                end
            end
        end
    end

    for batch in 1:batch_size
        for t in 1:out_seq_len
            for out_ch in 1:out_channels
                result[t, out_ch, batch] += b[out_ch]
            end
        end
    end

    return σ.(result)
end

function maxpool(x::AbstractArray, k::NTuple{1,Int}, pad::NTuple{2,Int}, stride::NTuple{1,Int})
    seq_len, features, batch_size = size(x)
    @assert pad[1] == 0 && pad[2] == 0
    out_seq_len = div(seq_len, k[1])
    result = zeros(eltype(x), out_seq_len, features, batch_size)

    for t in 1:out_seq_len
        t_start = (t - 1) * stride[1] + 1
        t_end = min(t_start + k[1] - 1, seq_len)
        for batch in 1:batch_size
            for f in 1:features
                max_val = maximum(view(x, t_start:t_end, f, batch))
                result[t, f, batch] = max_val
            end
        end
    end

    return result
end


function flatten(x::AbstractArray)
    return reshape(x, :, size(x)[end])
end