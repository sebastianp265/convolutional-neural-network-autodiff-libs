function visit!(node::GraphNode, visited::IdSet{GraphNode}, order::Vector{GraphNode})
    if node ∉ visited
        push!(visited, node)
        push!(order, node)
    end
    return nothing
end

function visit!(node::Operator, visited::IdSet{GraphNode}, order::Vector{GraphNode})
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
    visited = IdSet{GraphNode}()
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

is_output_scalar(::GraphNode{T}) where {T} = T <: Number

function gradient!(f::Function, args...)
    root::GraphNode = f(args...)
    # TODO: Check if necessary?
    if root isa Constant
        return zeros(length(args))
    end
    @assert is_output_scalar(root) "Function return value must be a scalar"

    root.gradient = one(eltype(root))

    order = topological_sort(root)
    compute!(order)

    for node in reverse(order)
        if node isa Operator
            grads = diff(node, [input.output for input in node.inputs]..., node.gradient)
            for (input, grad) in zip(node.inputs, grads)
                if !isa(input, Constant)
                    if isnothing(input.gradient)
                        input.gradient = grad
                    else
                        input.gradient += grad
                    end
                end
            end
        end
    end

    return map_args_to_gradient_result(args)
end

function map_args_to_gradient_result(args)
    return map(map_single_arg, args)
end

function map_single_arg(arg)
    if arg isa Variable
        return arg.gradient
    elseif arg isa Tuple
        return map(map_single_arg, arg)
    elseif isstructtype(typeof(arg))
        names = fieldnames(typeof(arg))
        if length(names) == 0
            return nothing
        end
        vals = map(name -> getfield(arg, name), names)
        mapped = map(map_single_arg, vals)

        return NamedTuple{names}(mapped)
    else
        return nothing
    end
end

