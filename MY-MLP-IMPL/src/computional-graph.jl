abstract type GraphNode{T} end
abstract type Operator{F,T} <: GraphNode{T} end

import Base: eltype

eltype(::GraphNode{T}) where {T} = eltype(T)

struct Constant{T} <: GraphNode{T}
    output::T
    name::Union{String,Nothing}
    Constant(output::T) where {T} = new{T}(output, nothing)
end

mutable struct Variable{T} <: GraphNode{T}
    output::T
    gradient::Any
    name::Union{String,Nothing}
    Variable(output::T, name::Union{String,Nothing}) where {T} = new{T}(output, nothing, name)
    Variable(output::T) where {T} = new{T}(output, nothing, nothing)
end

mutable struct ScalarOperator{F,T} <: Operator{F,T}
    inputs::Tuple{Vararg{GraphNode}}
    output::Union{T,Nothing}
    gradient::Any
end

# TODO: Refactor
function ScalarOperator(f::F, input::GraphNode{T}) where {F,T}
    ScalarOperator(f, input, false)
end

function ScalarOperator(::F, input::GraphNode{T}, is_output_eltype) where {F,T}
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
    gradient::Any
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

throw_arg_error(T1, T2) = throw(ArgumentError("Unsupported operation: unable to promote $T1 and $T2"))


function output_type(T1, T2)
    if T1 == T2
        return T1
    elseif !(T1 <: AbstractArray) && !(T2 <: AbstractArray)
        type = promote_type(T1, T2)
        if type === Any
            throw_arg_error(T1, T2)
        end
        return type
    elseif T1 <: AbstractVector && T2 <: AbstractVector
        type = promote_type(eltype(T1), eltype(T2))
        if type === Any
            throw_arg_error(T1, T2)
        end
        return Vector{type}
    elseif T2 <: AbstractMatrix && T2 <: AbstractMatrix
        type = promote_type(eltype(T1), eltype(T2))
        if type === Any
            throw_arg_error(T1, T2)
        end
        return Matrix{type}
    elseif T1 <: AbstractVector && T2 <: AbstractMatrix ||
           T1 <: AbstractMatrix && T2 <: AbstractVector
        type = promote_type(eltype(T1), eltype(T2))
        if type === Any
            throw_arg_error(T1, T2)
        end
        return Matrix{type}
    elseif T1 <: AbstractMatrix || T2 <: AbstractMatrix
        type = promote_type(eltype(T1), eltype(T2))
        if type === Any
            throw_arg_error(T1, T2)
        end
        return Matrix{type}
    elseif T1 <: AbstractVector || T2 <: AbstractVector
        type = promote_type(eltype(T1), eltype(T2))
        if type === Any
            throw_arg_error(T1, T2)
        end
        return Vector{type}
    else
        throw_arg_error(T1, T2)
    end
end

import Base: ==

==(x::Constant{X}, y::Constant{Y}) where {X,Y} =
    X === Y && x.output == y.output

==(x::Variable{X}, y::Variable{Y}) where {X,Y} =
    X === Y && x.output == y.output && x.gradient == y.gradient

==(x::Operator{FX,X}, y::Operator{FY,Y}) where {FX,FY,X,Y} =
    FX === FY && X === Y &&
    x.output == y.output &&
    x.gradient == y.gradient &&
    all([a == b for (a, b) in zip(x.inputs, y.inputs)])

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
    elseif s == (1,)
        return [sum(g)]
        # TODO: Invesitgate
    elseif s == ()
        return zero(eltype(g))
    else
        @show g s
        error("unsupported operation")
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
*(x, y) = ScalarOperator(*, x, promote_node(y))
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

# TODO: Add proper handling
sum(x::GraphNode; dims=1) = ScalarOperator(sum, x, true)
diff(::ScalarOperator{typeof(sum),T}, x, g) where {T} = tuple(g .* ones(T, size(x)))

sin(x::GraphNode) = ScalarOperator(sin, x)
diff(::BroadcastedOperator{typeof(sin)}, x, g) = tuple(g .* cos.(x))

# TODO: Maybe remove unnecessary scalars
log(x::GraphNode) = ScalarOperator(log, x)
diff(::BroadcastedOperator{typeof(log)}, x, g) = tuple(g ./ x)

mean(x::GraphNode) = ScalarOperator(mean, x, true)
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

# Show method for debugging

import Base: show

function show(io::IO, op::GraphNode)
    print(io, to_string(op))
end

function to_string(op::Operator{F}) where {F}
    f_name = if op isa BroadcastedOperator
        "."
    else
        ""
    end * string(F.instance)
    return f_name * "($(
        join(
            [to_string(input) for input in op.inputs],
            ", "
        )
        ))"
end

function to_string(node::GraphNode)
    if node isa Constant
        return node.name === nothing ? string(node.output) : node.name * " size = $(size(node.output))"
    elseif node isa Variable
        return node.name === nothing ? "?" : node.name * " size = $(size(node.output))"
    else
        return "<?>"
    end
end
