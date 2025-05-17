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
    # Reshape gradient back to original matrix shape
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
