abstract type GraphNode end
abstract type Operator <: GraphNode end

struct Constant{T} <: GraphNode
    output::T
end

mutable struct Variable{T} <: GraphNode
    output::T
end

mutable struct ScalarOperator{F} <: Operator
    inputs::Tuple{Vararg{GraphNode}}
    output::Any
    ScalarOperator(::F, args::GraphNode...) where {F} = new{F}(args, nothing)
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs::Tuple{Vararg{GraphNode}}
    output::Any
    BroadcastedOperator(::F, args::GraphNode...) where {F} = new{F}(args, nothing)
end

function visit!(node::GraphNode, visited::Set{GraphNode}, order::Vector{GraphNode})
    if node ∉ visited
        push!(visited, node)
        push!(order, node)
    end
    return nothing
end

function visit!(node::Operator, visited::Set{GraphNode}, order::Vector{GraphNode})
    if node ∉ visited
        push!(visited, node)
        for input in node.inputs
            visit!(input, visited, order)
        end
        push!(order, node)
    end
    return nothing
end

function topological_sort(root::GraphNode)
    visited = Set{GraphNode}()
    order = Vector{GraphNode}()
    visit!(root, visited, order)
    return order
end

compute!(::Constant) = nothing
compute!(::Variable) = nothing
compute!(node::Operator) = node.output = compute!(node, [input.output for input in node.inputs]...)

function compute!(compute_order::Vector{GraphNode})
    for node in compute_order
        compute!(node)
    end

    return last(compute_order).output
end

# compute! overloading

compute!(::ScalarOperator{typeof(+)}, x, y) = x + y
compute!(::ScalarOperator{typeof(-)}, x, y) = x - y
compute!(::ScalarOperator{typeof(-)}, x) = -x
compute!(::ScalarOperator{typeof(*)}, x, y) = x * y
compute!(::ScalarOperator{typeof(/)}, x, y) = x / y
compute!(::ScalarOperator{typeof(^)}, x, y) = x^y
compute!(::ScalarOperator{typeof(sum)}, x) = sum(x)
compute!(::ScalarOperator{typeof(sin)}, x) = sin(x)

compute!(::BroadcastedOperator{typeof(+)}, x, y) = x .+ y
compute!(::BroadcastedOperator{typeof(-)}, x, y) = x .- y
compute!(::BroadcastedOperator{typeof(*)}, x, y) = x .* y
compute!(::BroadcastedOperator{typeof(/)}, x, y) = x ./ y
compute!(::BroadcastedOperator{typeof(^)}, x, y) = x .^ y
compute!(::BroadcastedOperator{typeof(sum)}, x) = sum.(x)
compute!(::BroadcastedOperator{typeof(sin)}, x) = sin.(x)

# Base methods overloading

promote_node(x::Real) = Constant(x)

import Base: +, -, *, /, ^, sum, sin, log

+(x::GraphNode, y::GraphNode) = ScalarOperator(+, x, y)
+(x::Real, y::GraphNode) = ScalarOperator(+, promote_node(x), y)
+(x::GraphNode, y::Real) = ScalarOperator(+, x, promote_node(y))
Base.Broadcast.broadcasted(op::typeof(+), x::GraphNode, y::GraphNode) = BroadcastedOperator(op, x, y)
Base.Broadcast.broadcasted(op::typeof(+), x::Real, y::GraphNode) = BroadcastedOperator(op, promote_node(x), y)
Base.Broadcast.broadcasted(op::typeof(+), x::GraphNode, y::Real) = BroadcastedOperator(op, x, promote_node(y))

-(x::GraphNode, y::GraphNode) = ScalarOperator(-, x, y)
-(x::Real, y::GraphNode) = ScalarOperator(-, promote_node(x), y)
-(x::GraphNode, y::Real) = ScalarOperator(-, x, promote_node(y))
-(x::GraphNode) = ScalarOperator(-, x)
Base.Broadcast.broadcasted(op::typeof(-), x::GraphNode, y::GraphNode) = BroadcastedOperator(op, x, y)
Base.Broadcast.broadcasted(op::typeof(-), x::Real, y::GraphNode) = BroadcastedOperator(op, promote_node(x), y)
Base.Broadcast.broadcasted(op::typeof(-), x::GraphNode, y::Real) = BroadcastedOperator(op, x, promote_node(y))

*(x::GraphNode, y::GraphNode) = ScalarOperator(*, x, y)
*(x::Real, y::GraphNode) = ScalarOperator(*, promote_node(x), y)
*(x::GraphNode, y::Real) = ScalarOperator(*, x, promote_node(y))
Base.Broadcast.broadcasted(op::typeof(*), x::GraphNode, y::GraphNode) = BroadcastedOperator(op, x, y)
Base.Broadcast.broadcasted(op::typeof(*), x::Real, y::GraphNode) = BroadcastedOperator(op, promote_node(x), y)
Base.Broadcast.broadcasted(op::typeof(*), x::GraphNode, y::Real) = BroadcastedOperator(op, x, promote_node(y))

/(x::GraphNode, y::GraphNode) = ScalarOperator(/, x, y)
/(x::Real, y::GraphNode) = ScalarOperator(/, promote_node(x), y)
/(x::GraphNode, y::Real) = ScalarOperator(/, x, promote_node(y))
Base.Broadcast.broadcasted(op::typeof(/), x::GraphNode, y::GraphNode) = BroadcastedOperator(op, x, y)
Base.Broadcast.broadcasted(op::typeof(/), x::Real, y::GraphNode) = BroadcastedOperator(op, promote_node(x), y)
Base.Broadcast.broadcasted(op::typeof(/), x::GraphNode, y::Real) = BroadcastedOperator(op, x, promote_node(y))

^(x::GraphNode, y::GraphNode) = ScalarOperator(^, x, y)
^(x::Real, y::GraphNode) = ScalarOperator(^, promote_node(x), y)
^(x::GraphNode, y::Real) = ScalarOperator(^, x, promote_node(y))
Base.Broadcast.broadcasted(op::typeof(^), x::GraphNode, y::GraphNode) = BroadcastedOperator(op, x, y)
Base.Broadcast.broadcasted(op::typeof(^), x::Real, y::GraphNode) = BroadcastedOperator(op, promote_node(x), y)
Base.Broadcast.broadcasted(op::typeof(^), x::GraphNode, y::Real) = BroadcastedOperator(op, x, promote_node(y))
# This broadcasted operator overload is required because of Julia optimizations, 
# x .^ 2 calls optimized implementation
Base.broadcasted(::typeof(Base.literal_pow), ::Function, x::Variable, y::Val) =
    BroadcastedOperator(^, x, promote_node(typeof(y).parameters[1]))

# TODO: Add proper handling
sum(x::GraphNode; dims=1) = ScalarOperator(sum, x)
Base.Broadcast.broadcasted(op::typeof(sum), x::GraphNode) = BroadcastedOperator(op, x)

sin(x::GraphNode) = ScalarOperator(sin, x)
Base.Broadcast.broadcasted(op::typeof(sin), x::GraphNode) = BroadcastedOperator(op, x)

log(x::GraphNode) = ScalarOperator(log, x)
Base.Broadcast.broadcasted(op::typeof(log), x::GraphNode) = BroadcastedOperator(op, x)

import Statistics: mean

mean(x::GraphNode) = ScalarOperator(mean, x)
Base.Broadcast.broadcasted(op::typeof(mean), x::GraphNode) = BroadcastedOperator(op, x)



# TODO: Is needed?
length(x::Variable) = length(x.output)

# TODO: Is needed?
Base.broadcastable(x::GraphNode) = x

# Overloading commonly used functions

Base.eltype(x::GraphNode) = eltype(x.output)
