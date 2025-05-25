abstract type GraphNode{T} end
abstract type Operator{F,T} <: GraphNode{T} end

import Base: eltype, size

eltype(::GraphNode{T}) where {T} = eltype(T)
size(x::GraphNode, dims...) = size(x.output, dims...)

struct Constant{T} <: GraphNode{T}
    output::T
    Constant(output::T) where {T} = new{T}(output)
end

mutable struct Variable{T} <: GraphNode{T}
    output::T
    gradient::Any # TODO: Think about not wrapping in any, during creation it can be exact type
    Variable(output::T) where {T} = new{T}(output, nothing)
end

mutable struct ScalarOperator{F,T} <: Operator{F,T}
    inputs::Tuple{Vararg{GraphNode}}
    output::T
    gradient::Any
    ScalarOperator(f::F, inputs::GraphNode...; kwargs...) where {F} = begin
        output = f(map(i -> i.output, inputs)...; kwargs...)
        new{F,typeof(output)}(inputs, output, nothing)
    end
end

mutable struct BroadcastedOperator{F,T} <: Operator{F,T}
    inputs::Tuple{Vararg{GraphNode}}
    output::T
    gradient::Any
    BroadcastedOperator(f::F, inputs::GraphNode...) where {F} = begin
        output = f.(map(i -> i.output, inputs)...)
        new{F,typeof(output)}(inputs, output, nothing)
    end
end

# Base methods overloading

function clean_constant(x)
    if x isa Base.Broadcast.Broadcasted
        return Base.materialize(x)
    else
        return x
    end
end

promote_node(x) = Constant(clean_constant(x))

Base.broadcasted(f, x::GraphNode) = BroadcastedOperator(f, x)
Base.broadcasted(f, x::GraphNode, y::GraphNode) = BroadcastedOperator(f, x, y)
Base.broadcasted(f, x, y::GraphNode) = BroadcastedOperator(f, promote_node(x), y)
Base.broadcasted(f, x::GraphNode, y) = BroadcastedOperator(f, x, promote_node(y))

# This broadcasted operator overload is required because of Julia optimizations, 
# for example: x .^ 2 calls optimized implementation
Base.broadcasted(::typeof(Base.literal_pow), ::Function, x::Variable, y::Val) =
    BroadcastedOperator(^, x, promote_node(typeof(y).parameters[1]))

import Base: +, -, *, /, ^

+(x::GraphNode, y::GraphNode) = ScalarOperator(+, x, y)
+(x, y::GraphNode) = ScalarOperator(+, promote_node(x), y)
+(x::GraphNode, y) = ScalarOperator(+, x, promote_node(y))
+(x::GraphNode) = ScalarOperator(+, x)
diff(::ScalarOperator{typeof(+)}, x, y, g) = tuple(g, g)
diff(::ScalarOperator{typeof(+)}, x, g) = tuple(g)
diff(::BroadcastedOperator{typeof(+)}, x, y, g) = tuple(acc_to_size(g, size(x)), acc_to_size(g, size(y)))

acc_to_size(g, s) = begin
    if size(g) == s
        return g
    elseif length(s) == 1
        res = sum(g; dims=2)
        return if size(res)[2] == 1
            vec(res)
        else
            #TODO: Investigate if needed
            res
        end
    elseif s == ()
        return zero(eltype(g))
    else
        error("unsupported operation size(g)=$(size(g)) s=$s")
    end
end

-(x::GraphNode, y::GraphNode) = ScalarOperator(-, x, y)
-(x, y::GraphNode) = ScalarOperator(-, promote_node(x), y)
-(x::GraphNode, y) = ScalarOperator(-, x, promote_node(y))
-(x::GraphNode) = ScalarOperator(-, x)
diff(::Operator{typeof(-)}, x, y, g) = tuple(g, -g)
diff(::Operator{typeof(-)}, x, g) = tuple(-g)

*(x::GraphNode, y::GraphNode) = ScalarOperator(*, x, y)
*(x, y::GraphNode) = ScalarOperator(*, promote_node(x), y)
*(x::GraphNode, y) = ScalarOperator(*, x, promote_node(y))
diff(::ScalarOperator{typeof(*)}, x, y, g) = tuple(g * y', x' * g)
diff(::BroadcastedOperator{typeof(*)}, x, y, g) = tuple(g .* y, g .* x)

/(x::GraphNode, y::GraphNode) = ScalarOperator(/, x, y)
/(x, y::GraphNode) = ScalarOperator(/, promote_node(x), y)
/(x::GraphNode, y) = ScalarOperator(/, x, promote_node(y))
diff(::BroadcastedOperator{typeof(/)}, x, y, g) = tuple(g ./ y, -g .* x ./ (y .* y))

^(x::GraphNode, y::GraphNode) = ScalarOperator(^, x, y)
^(x, y::GraphNode) = ScalarOperator(^, promote_node(x), y)
^(x::GraphNode, y) = ScalarOperator(^, x, promote_node(y))
diff(::BroadcastedOperator{typeof(^)}, x, y, g) = tuple(g .* y .* x .^ (y .- 1), g .* x .^ y .* log.(x))

import Base: sum, sin, log, max
import Statistics: mean

sum(x::GraphNode; dims...) = ScalarOperator(sum, x; dims...)
diff(::ScalarOperator{typeof(sum),T}, x, g) where {T} = tuple(g .* ones(T, size(x)))

sin(x::GraphNode) = ScalarOperator(sin, x)
diff(::BroadcastedOperator{typeof(sin)}, x, g) = tuple(g .* cos.(x))

log(x::GraphNode) = ScalarOperator(log, x)
diff(::BroadcastedOperator{typeof(log)}, x, g) = tuple(g ./ x)

mean(x::GraphNode) = ScalarOperator(mean, x)
diff(::ScalarOperator{typeof(mean),T}, x, g) where {T} = tuple(fill(g * one(eltype(T)) / length(x), size(x)))

diff(::BroadcastedOperator{typeof(sigmoid)}, x, g) = begin
    sigm = sigmoid.(x)
    grad = sigm .* (1 .- sigm)
    return tuple(g .* grad)
end
diff(::BroadcastedOperator{typeof(relu)}, x, g) = begin
    grad = x .> 0
    return tuple(g .* grad)
end

diff(::BroadcastedOperator{typeof(xlogy)}, x, y, g) = begin
    dx = g .* log.(y)
    dy = g .* (x ./ y)
    return (dx, dy)
end

to_3d(x::GraphNode, s::NTuple{2,Integer}) = ScalarOperator(to_3d, x, promote_node(s))
diff(::ScalarOperator{typeof(to_3d)}, x, s, g) = begin
    tuple(reshape(g, size(x)), nothing)
end

gather(x::GraphNode, y) = ScalarOperator(gather, x, promote_node(y))
diff(::ScalarOperator{typeof(gather)}, W, idxs, g) = begin
    grad_W = zeros(eltype(g), size(W))
    for (j, idx) in enumerate(idxs)
        grad_W[:, idx] .+= g[:, j]
    end
    return (grad_W, nothing)
end

import Base: permutedims

permutedims(x::GraphNode, perm) = ScalarOperator(permutedims, x, promote_node(perm))
diff(::ScalarOperator{typeof(permutedims)}, x, perm, g) = tuple(permutedims(g, invperm(perm)), nothing)

conv1d(x::GraphNode, y::GraphNode, z::GraphNode, σ, stride, pad, dilation, groups) =
    ScalarOperator(conv1d, x, y, z, promote_node(σ),
        promote_node(stride), promote_node(pad), promote_node(dilation), promote_node(groups))

diff(::ScalarOperator{typeof(conv1d)}, W, x, b, σ, stride, pad, dilation, groups, g) = begin
    #print(g)
    kernel_size, in_channels_per_group, out_channels = size(W)
    seq_len, features, batch_size = size(x)
    grad_W = zeros(eltype(g), size(W))
    grad_x = zeros(eltype(g), size(x))
    grad_b = zeros(eltype(g), size(b))

    if pad[1] > 0 || pad[2] > 0
        padded_x = zeros(eltype(x), seq_len + pad[1] + pad[2], features, batch_size)
        padded_x[pad[1]+1:pad[1]+seq_len, :, :] = x
        x = padded_x
        seq_len = size(x, 1)

        padded_grad_x = zeros(eltype(grad_x), seq_len, features, batch_size)
        grad_x = padded_grad_x
    end

    out_seq_len = div(seq_len - (kernel_size - 1) * dilation[1] - 1, stride[1]) + 1

    result = conv1d(W, x, b, identity, stride, pad, dilation, groups)

    if σ != identity
        g = g .* σ.(result)
    end

    for batch in 1:batch_size
        for t in 1:out_seq_len
            t_start = (t - 1) * stride[1] + 1
            for out_ch in 1:out_channels
                grad_b[out_ch] += g[t, out_ch, batch]

                for g_idx in 1:groups
                    in_ch_start = (g_idx - 1) * in_channels_per_group + 1
                    in_ch_end = g_idx * in_channels_per_group
                    for in_ch_offset in 1:in_channels_per_group
                        in_ch = in_ch_start + in_ch_offset - 1
                        for k in 1:kernel_size
                            t_pos = t_start + (k - 1) * dilation[1]
                            grad_W[k, in_ch_offset, out_ch] += x[t_pos, in_ch, batch] * g[t, out_ch, batch]
                            grad_x[t_pos, in_ch, batch] += W[k, in_ch_offset, out_ch] * g[t, out_ch, batch]
                        end
                    end
                end
            end
        end
    end
    if pad[1] > 0 || pad[2] > 0
        grad_x = grad_x[pad[1]+1:pad[1]+seq_len-pad[2], :, :]
    end

    return (grad_W, grad_x, grad_b, nothing, nothing, nothing, nothing, nothing)
end


maxpool(x::GraphNode, k::NTuple{1,Int}, pad::NTuple{2,Int}, stride::NTuple{1,Int}) =
    ScalarOperator(maxpool, x, promote_node(k), promote_node(pad), promote_node(stride))

diff(::ScalarOperator{typeof(maxpool)}, x, k, pad, stride, g) = begin
    seq_len, features, batch_size = size(x)
    result_g = zeros(eltype(x), seq_len, features, batch_size)
    @assert pad[1] == 0 && pad[2] == 0
    out_seq_len = div(seq_len, k[1])

    for batch in 1:batch_size
        for seq in 1:out_seq_len
            t_start = (seq - 1) * stride[1] + 1
            t_end = min(t_start + k[1] - 1, seq_len)
            for f in 1:features
                max_idx = argmax(x[t_start:t_end, f, batch])
                result_g[t_start+max_idx-1, f, batch] = g[seq, f, batch]
            end
        end
    end

    return (result_g, nothing, nothing, nothing)
end

flatten(x::GraphNode) = ScalarOperator(flatten, x)
diff(::ScalarOperator{typeof(flatten)}, x, g) = begin
    return (reshape(g, size(x)),)
end