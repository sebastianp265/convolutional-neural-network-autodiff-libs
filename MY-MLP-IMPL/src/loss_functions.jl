import Statistics: mean

function crossentropy(ŷ, y; dims=1, ϵ=eps(eltype(ŷ)), agg=mean)
    agg(-sum(y .* log.(ŷ .+ ϵ); dims))
end

function binarycrossentropy(ŷ, y; agg=mean, ϵ=eps(eltype(ŷ)))
    agg(@.(-y * log(ŷ + ϵ) - (1 - y) * log(1 - ŷ + ϵ)))
end


