import Statistics: mean

function crossentropy(ŷ, y; dims=1, ϵ=eps(eltype(ŷ)), agg=mean)
    agg(-sum(y .* log.(ŷ .+ ϵ); dims))
end

function binarycrossentropy(ŷ, y; agg=mean, ϵ=eps(eltype(ŷ)))
    agg(@.(-xlogy(y, ŷ + ϵ) - xlogy(1 - y, 1 - ŷ + ϵ)))
end

function xlogy(x, y)
    result = x * log(y)
    ifelse(iszero(x), zero(result), result)
end

