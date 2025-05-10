abstract type AbstractRule end

struct Adam <: AbstractRule
    eta::Float64
    beta::Tuple{Float64,Float64}
    epsilon::Float64
    Adam(; eta=0.001, beta=(0.9, 0.999), epsilon=1.0e-8) = new(eta, beta, epsilon)
end

function Base.copy(opt::Adam)
    return Adam(eta=opt.eta, beta=opt.beta, epsilon=opt.epsilon)
end

function Base.show(io::IO, opt::Adam)
    print(io, "Adam(eta=$(opt.eta), beta=$(opt.beta), epsilon=$(opt.epsilon))")
end

mutable struct Leaf
    rule::AbstractRule
    state::Tuple
end

function setup(rule::AbstractRule, model)
    return _setup(rule, model)
end

function _setup(rule::AbstractRule, node)
    if node isa NamedTuple || node isa Tuple
        return map(v -> _setup(rule, v), node)
    elseif node isa Variable
        s = size(node.output)
        element_type = eltype(node.output)
        return Leaf(copy(rule), (
            zeros(element_type, s),
            zeros(element_type, s),
            map(e -> convert(element_type, e), rule.beta)
        ))
    elseif isstructtype(typeof(node))
        names = fieldnames(typeof(node))
        if length(names) == 0
            return tuple()
        end
        vals = map(name -> getfield(node, name), names)
        mapped = map(v -> _setup(rule, v), vals)

        return NamedTuple{names}(mapped)
    else
        return tuple()
    end
end

function update!(state, model, gradients)
    _update!(state, model, gradients)
end

function _update!(opt_state, updateable_state, gradient_state)
    assert_types_correct(typeof(opt_state), typeof(updateable_state), typeof(gradient_state))

    if typeof(updateable_state) <: Function
        return
    elseif updateable_state isa Variable
        optimize!(opt_state, updateable_state.output, gradient_state) # TODO: Refactor, remove maybe Variable wrapper and check performance diff?
    elseif isstructtype(typeof(updateable_state))
        for fn in fieldnames(typeof(updateable_state))
            _update!(
                getfield(opt_state, fn),
                getfield(updateable_state, fn),
                getfield(gradient_state, fn)
            )
        end
    else
        error("Unsupported type: $updateable_state")
    end

end

# TODO: Delete after final impl
function assert_types_correct(opt_state_t, updateable_state_t, gradient_state_t)
    if (updateable_state_t <: Function && opt_state_t === Tuple{} && isnothing(gradient_state_t)) ||
       (isstructtype(updateable_state_t) && opt_state_t === NamedTuple && gradient_state_t === NamedTuple) ||
       (updateable_state_t === Tuple && opt_state_t === Tuple && gradient_state_t === Tuple) ||
       return
    else
        error("Unexpected type combination opt_state_t=$opt_state_t, updateable_state_t=$updateable_state_t, gradient_state_t=$gradient_state_t")
    end
end

function optimize!(leaf::Leaf, x::AbstractArray{T}, gradient::AbstractArray{T}) where {T}
    opt = leaf.rule

    ƞ, β, ϵ = T(opt.eta), T.(opt.beta), _eps(T, opt.epsilon)
    mt, vt, βt = leaf.state

    mt .= β[1] .* mt .+ (1 .- β[1]) .* gradient
    vt .= β[2] .* vt .+ (1 .- β[2]) .* abs2.(gradient)
    dx = mt / (1 - βt[1]) ./ (sqrt.(vt / (1 - βt[2])) .+ ϵ) * ƞ

    x .= x .- dx

    leaf.state = (mt, vt, βt .* β)
end

_eps(T::Type{<:Number}, e) = real(float(T))(e)
_eps(T::Type{Float16}, e) = e == 0 ? T(0) : max(T(1e-7), T(e))
