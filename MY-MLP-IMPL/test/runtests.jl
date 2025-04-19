using Test
using MYMLP
using Statistics
import Base: ==
import Flux

==(x::Constant{X}, y::Constant{Y}) where {X,Y} =
    X === Y && x.output == y.output

==(x::Variable{X}, y::Variable{Y}) where {X,Y} =
    X === Y && x.output == y.output && x.gradient == y.gradient

==(x::Operator{FX,X}, y::Operator{FY,Y}) where {FX,FY,X,Y} =
    FX === FY && X === Y &&
    x.output == y.output &&
    x.gradient == y.gradient &&
    all([a == b for (a, b) in zip(x.inputs, y.inputs)])


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
    x_int = Variable([i^2 for i in 1:4])
    x_float = Variable(Float32.([i^2 for i in 1:4]))

    @test typeof(sum.(x_int)) == BroadcastedOperator{typeof(sum),Vector{Int64}}
    @test typeof(sum.(x_float)) == BroadcastedOperator{typeof(sum),Vector{Float32}}
    @test typeof(sum(x_int)) == ScalarOperator{typeof(sum),Int64}
    @test typeof(sum(x_float)) == ScalarOperator{typeof(sum),Float32}

    two = 2

    @test typeof(x_int * two) == ScalarOperator{typeof(*),Vector{Int64}}
    @test typeof(x_float * two) == ScalarOperator{typeof(*),Vector{Float32}}
    @test typeof(x_int / two) == ScalarOperator{typeof(/),Vector{Int64}}
    @test typeof(x_float / two) == ScalarOperator{typeof(/),Vector{Float32}}
    @test typeof(x_int .^ two) == BroadcastedOperator{typeof(^),Vector{Int64}}
    @test typeof(x_float .^ two) == BroadcastedOperator{typeof(^),Vector{Float32}}
    @test typeof(x_int .+ two) == BroadcastedOperator{typeof(+),Vector{Int64}}
    @test typeof(x_float .+ two) == BroadcastedOperator{typeof(+),Vector{Float32}}

    y = Variable(Float64.([i^2 for i in 1:10]))

    @test typeof(y .* x_int) == BroadcastedOperator{typeof(*),Vector{Float64}}
    @test typeof(y .* x_float) == BroadcastedOperator{typeof(*),Vector{Float64}}
end

@testset "Computional Graph Creation" begin
    x = Variable(Float32.([i^3 for i in 1:3]))
    y = Variable([i for i in 1:3])

    expr = x .^ 2 - y .^ 3 .- 1
    @test expr == BroadcastedOperator(
        -,
        ScalarOperator(-,
            BroadcastedOperator(
                ^,
                Variable(Float32.([i^3 for i in 1:3])),
                Constant(2)
            ),
            BroadcastedOperator(
                ^,
                Variable([i for i in 1:3]),
                Constant(3)
            )
        ),
        Constant(1)
    )

    traverse_order = topological_sort(expr)
    @test map(node -> typeof(node), traverse_order) == [
        Variable{Vector{Float32}},
        Constant{Int64},
        BroadcastedOperator{typeof(^),Vector{Float32}},
        Variable{Vector{Int64}},
        Constant{Int64},
        BroadcastedOperator{typeof(^),Vector{Int64}},
        ScalarOperator{typeof(-),Vector{Float32}},
        Constant{Int64},
        BroadcastedOperator{typeof(-),Vector{Float32}}
    ]

    output = compute!(traverse_order)
    @test output == [1^2 - 1^3 - 1, 8^2 - 2^3 - 1, 27^2 - 3^3 - 1]

    # Example of an function that needs to be supported, definition taken from Flux.jl documentation
    function crossentropy(ŷ, y; dims=1, ϵ=eps(eltype(ŷ)), agg=mean)
        agg(-sum(y .* log.(ŷ .+ ϵ); dims))
    end

    @test eltype(x) == Float32
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
                        [1, 2, 3]
                    ),
                    BroadcastedOperator(
                        log,
                        BroadcastedOperator(
                            +,
                            Variable(Float32.([1, 8, 27])),
                            Constant(eps(Float32)),
                        ),
                    )
                ),
                true,
            ),
        ),
        true,
    )

    x_unwraped = Float32.([i^3 for i in 1:3])
    y_unwraped = [i for i in 1:3]

    expected_crossentropy_evaluation = Flux.Losses.crossentropy(x_unwraped, y_unwraped)

    @test evaluate!(expr) == expected_crossentropy_evaluation

    flux_expr = Flux.Losses.crossentropy(x, y)
    @test evaluate!(flux_expr) == expected_crossentropy_evaluation
end

