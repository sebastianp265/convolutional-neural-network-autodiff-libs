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

clear_output!(node::Operator) = node.output = nothing
clear_output!(node::Variable) = node.output = nothing
clear_output!(::GraphNode) = nothing

function clear_output!(compute_order::Vector{GraphNode})
    for node in compute_order
        clear_output!(node)
    end
end

is_output_scalar(::GraphNode{T}) where {T} = T <: Number

reset_gradient!(node::Variable) = node.gradient = nothing
reset_gradient!(node::Operator) = node.gradient = nothing
reset_gradient!(::Constant) = nothing
function reset_gradient!(compute_order::Vector{GraphNode})
    for node in compute_order
        reset_gradient!(node)
    end
end

function gradient!(f::Function, args...)
    root = f(args...)
    # TODO: Check if necessary?
    if root isa Constant
        return zeros(length(args))
    end
    @assert root isa GraphNode
    if !is_output_scalar(root)
        error("Function return value must be a scalar")
    end

    order = topological_sort(root)
    reset_gradient!(order)
    root.gradient = one(eltype(root))

    for node in reverse(order)
        if node isa Operator
            grads = diff(node, [input.output for input in node.inputs]..., node.gradient)
            for (input, grad) in zip(node.inputs, grads)
                if !isa(input, Constant)
                    if isnothing(input.gradient)
                        input.gradient = grad
                    else
                        # TODO: Add broadcasted?
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
        mapped = map(map_single_arg, arg)

        if all(isnothing, mapped)
            return nothing
        else
            return mapped
        end
    elseif isstructtype(typeof(arg))
        names = fieldnames(typeof(arg))
        if length(names) == 0
            return nothing
        end
        vals = map(name -> getfield(arg, name), names)
        mapped = map(map_single_arg, vals)

        if all(isnothing, mapped)
            return nothing
        else
            return NamedTuple{names}(mapped)
        end
    elseif typeof(arg) <: Number
        return nothing
    else
        error("$arg")
    end
end

