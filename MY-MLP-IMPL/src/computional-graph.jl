abstract type GraphNode{T} end
abstract type Operator{F,T} <: GraphNode{T} end

import Base: eltype

eltype(::GraphNode{T}) where {T} = eltype(T)

struct Constant{T} <: GraphNode{T}
    output::T
end

mutable struct Variable{T} <: GraphNode{T}
    output::T
    gradient::Union{T,Nothing}
    Variable(output::T) where {T} = new{T}(output, nothing)
end

mutable struct ScalarOperator{F,T} <: Operator{F,T}
    inputs::Tuple{Vararg{GraphNode}}
    output::Union{T,Nothing}
    gradient::Union{T,Nothing}
end

function ScalarOperator(::F, input::GraphNode{T}, is_output_eltype::Bool=false) where {F,T}
    type = if is_output_eltype
        eltype(T)
    else
        T
    end
    ScalarOperator{F,type}(tuple(input), nothing, nothing)
end

function ScalarOperator(::F, input1::GraphNode{T1}, input2::GraphNode{T2}) where {F,T1,T2}
    type = output_type(T1, T2)
    ScalarOperator{F,type}(tuple(input1, input2), nothing, nothing)
end

function ScalarOperator(::F, inputs::GraphNode...) where {F}
    error("Scalar operator '$F' is not defined on arguments: $inputs")
end

mutable struct BroadcastedOperator{F,T} <: Operator{F,T}
    inputs::Tuple{Vararg{GraphNode}}
    output::Union{T,Nothing}
    gradient::Union{T,Nothing}
end

function BroadcastedOperator(::F, input::GraphNode{T}) where {F,T}
    BroadcastedOperator{F,T}(tuple(input), nothing, nothing)
end

function BroadcastedOperator(::F, input1::GraphNode{T1}, input2::GraphNode{T2}) where {F,T1,T2}
    type = output_type(T1, T2)
    BroadcastedOperator{F,type}(tuple(input1, input2), nothing, nothing)
end

function BroadcastedOperator(::F, inputs::GraphNode...) where {F}
    error("Broadcasted operator '.$F' is not defined on arguments: $inputs")
end

function output_type(T1, T2)
    if T1 <: AbstractArray && !(T2 <: AbstractArray)
        return T1
    elseif T2 <: AbstractArray && !(T1 <: AbstractArray)
        return T2
    elseif T1 <: AbstractArray && T2 <: AbstractArray
        promote_el = promote_type(eltype(T1), eltype(T2))
        if promote_el === Any
            throw(ArgumentError("Unsupported operation: unable to promote $T1 and $T2"))
        end
        return Vector{promote_el}
    else
        promoted_type = promote_type(T1, T2)
        if promoted_type === Any
            throw(ArgumentError("Unsupported operation: unable to promote $T1 and $T2"))
        end
        return promoted_type
    end
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
    # TODO: Maybe change to IdSet?
    visited = Set{GraphNode}()
    order = Vector{GraphNode}()
    visit!(root, visited, order)
    return order
end

compute!(::Constant) = nothing
compute!(::Variable) = nothing
compute!(node::Operator) = node.output = compute!(node, [input.output for input in node.inputs]...)

compute!(::ScalarOperator{F}, x) where {F} = F.instance(x)
compute!(::ScalarOperator{F}, x, y) where {F} = F.instance(x, y)
compute!(::ScalarOperator{F}, x, y, z, args...) where {F} = error("Scalar operations on more than two arguments are disabled")
compute!(::BroadcastedOperator{F}, x) where {F} = F.instance.(x)
compute!(::BroadcastedOperator{F}, x, y) where {F} = F.instance.(x, y)
compute!(::BroadcastedOperator{F}, x, y, z, args...) where {F} = error("Broadcast operations on more than two arguments are disabled")

function compute!(compute_order::Vector{GraphNode})
    for node in compute_order
        compute!(node)
    end

    return last(compute_order).output
end

function evaluate!(root::GraphNode)
    order = topological_sort(root)
    return compute!(order)
end

diff(::BroadcastedOperator{typeof(+)}, x, y) = tuple(1, 1)
diff(::BroadcastedOperator{typeof(sin)}, x) = tuple(cos.(x))
diff(::BroadcastedOperator{typeof(*)}, x, y) = tuple(y, x)

function gradient(f, args...)
    root = f(args...)
    @assert root isa GraphNode "Function must create computional graph"

    root.gradient = 1.0
    order = topological_sort(root)

    for node in reverse(order)
        if node isa Operator
            gradients = diff(node, [input.output for input in node.inputs]...) .* node.gradient
            for (input, gradient) in zip(node.inputs, gradients)
                if !isa(input, Constant)
                    if isnothing(input.gradient)
                        input.gradient = gradient
                    else
                        input.gradient += gradient
                    end
                end
            end
        end
    end
end


# Base methods overloading

promote_node(x) = Constant(x)

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

-(x::GraphNode, y::GraphNode) = ScalarOperator(-, x, y)
-(x, y::GraphNode) = ScalarOperator(-, promote_node(x), y)
-(x::GraphNode, y) = ScalarOperator(-, x, promote_node(y))
-(x::GraphNode) = ScalarOperator(-, x)

*(x::GraphNode, y::GraphNode) = ScalarOperator(*, x, y)
*(x, y::GraphNode) = ScalarOperator(*, promote_node(x), y)
*(x, y) = ScalarOperator(*, x, promote_node(y))

/(x::GraphNode, y::GraphNode) = ScalarOperator(/, x, y)
/(x, y::GraphNode) = ScalarOperator(/, promote_node(x), y)
/(x::GraphNode, y) = ScalarOperator(/, x, promote_node(y))

^(x::GraphNode, y::GraphNode) = ScalarOperator(^, x, y)
^(x, y::GraphNode) = ScalarOperator(^, promote_node(x), y)
^(x::GraphNode, y) = ScalarOperator(^, x, promote_node(y))

import Base: sum, sin, log, max
import Statistics: mean

# TODO: Add proper handling
sum(x::GraphNode; dims=1) = ScalarOperator(sum, x, true)
sin(x::GraphNode) = ScalarOperator(sin, x)
log(x::GraphNode) = ScalarOperator(log, x)
mean(x::GraphNode) = ScalarOperator(mean, x, true)
