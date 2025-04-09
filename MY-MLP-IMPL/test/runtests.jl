using Test
using MYMLP
using Statistics
import Base: ==

==(a::Variable, b::Variable) = a.output == b.output
==(a::Constant, b::Constant) = a.output == b.output
==(a::ScalarOperator, b::ScalarOperator) = a.inputs == b.inputs && a.output == b.output
==(a::BroadcastedOperator, b::BroadcastedOperator) = a.inputs == b.inputs && a.output == b.output

@testset "Dense Layer Test" begin
    d = Dense(3, 2, relu)
    x = rand(Float32, 3)
    y = d(x)

    @test size(y) == (2,)

    @test all(y .>= 0)
end

@testset "Chain Test" begin
    c = Chain(
        Dense(3, 4, relu),
        Dense(4, 1, identity)
    )

    x = rand(Float32, 3)
    y = c(x)

    @test size(y) == (1,)
end

@testset "Computional Graph Types" begin
    x = Variable([i^2 for i in 1:10])
    two = 2

    @test typeof(x * two) == ScalarOperator{typeof(*)}
    @test typeof(x / two) == ScalarOperator{typeof(/)}
    @test typeof(x + two) == ScalarOperator{typeof(+)}
    @test typeof(x - two) == ScalarOperator{typeof(-)}
    @test typeof(x^two) == ScalarOperator{typeof(^)}
    @test typeof(x .+ two) == BroadcastedOperator{typeof(+)}
    @test typeof(sum.(x)) == BroadcastedOperator{typeof(sum)}
    @test typeof(sum(x)) == ScalarOperator{typeof(sum)}

    y = Variable([i^2 for i in 1:10])

    @test typeof(x * y) == ScalarOperator{typeof(*)}
    @test typeof(x .* y) == BroadcastedOperator{typeof(*)}
end

@testset "Computional Graph Creation" begin
    x = Variable(Float32.([i^3 for i in 1:3]))
    y = Variable(Float32.([i^2 for i in 1:3]))

    expr = x .^ 2 - y .^ 2 .- 1

    @test expr == BroadcastedOperator(
        -,
        ScalarOperator(-,
            BroadcastedOperator(
                ^,
                Variable([1, 8, 27]),
                Constant(2)
            ),
            BroadcastedOperator(
                ^,
                Variable([1, 4, 9]),
                Constant(2)
            )
        ),
        Constant(1)
    )

    traverse_order = topological_sort(expr)
    output = compute!(traverse_order)
    @test output == [-1, 8^2 - 4^2 - 1, 27^2 - 9^2 - 1]

    # Example of an function that needs to be supported, definition taken from Flux.jl documentation
    function crossentropy(ŷ, y; dims=1, ϵ=eps(eltype(ŷ)), agg=mean)
        agg(-sum(y .* log.(ŷ .+ ϵ); dims))
    end

    expr = crossentropy(x, y)
    @test expr == ScalarOperator(
        mean,
        ScalarOperator(
            -,
            ScalarOperator(
                sum,
                BroadcastedOperator(
                    *,
                    Variable(
                        [1, 4, 9]
                    ),
                    BroadcastedOperator(
                        log,
                        BroadcastedOperator(
                            +,
                            Variable([1, 8, 27]),
                            Constant(eps(Float32)),
                        ),
                    )
                ),
            ),
        ),
    )
end
