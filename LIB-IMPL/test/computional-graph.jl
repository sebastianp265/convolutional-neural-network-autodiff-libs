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
    @test typeof(x_int / two) == ScalarOperator{typeof(/),Vector{Float64}}
    @test typeof(x_float / two) == ScalarOperator{typeof(/),Vector{Float32}}
    @test typeof(x_int .^ two) == BroadcastedOperator{typeof(^),Vector{Int64}}
    @test typeof(x_float .^ two) == BroadcastedOperator{typeof(^),Vector{Float32}}
    @test typeof(x_int .+ two) == BroadcastedOperator{typeof(+),Vector{Int64}}
    @test typeof(x_float .+ two) == BroadcastedOperator{typeof(+),Vector{Float32}}

    y = Variable(Float64.([i^2 for i in 1:4]))

    @test typeof(y .* x_int) == BroadcastedOperator{typeof(*),Vector{Float64}}
    @test typeof(y .* x_float) == BroadcastedOperator{typeof(*),Vector{Float64}}
end

@testset "Computional Graph Creation" begin
    x = Variable(Float32.([i^3 for i in 1:3]))
    y = Variable([i for i in 1:3])

    @test eltype(x) == Float32
    @test eltype(y) == Int64

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

    @test expr.output == [1^2 - 1^3 - 1, 8^2 - 2^3 - 1, 27^2 - 3^3 - 1]
end

@testset "Computional Graph Computation" begin
    x_unwraped = Float32.([1 / i^2 for i in 1:3])
    y_unwraped = [-1 / i^2 for i in 1:3]

    expected_crossentropy_evaluation = Flux.Losses.binarycrossentropy(x_unwraped, y_unwraped)

    @test binarycrossentropy(x_unwraped, y_unwraped) == expected_crossentropy_evaluation

    x = Variable(Float32.([1 / i^2 for i in 1:3]))
    y = Variable(([-1 / i^2 for i in 1:3]))

    expr = binarycrossentropy(x, y)

    @test expr == ScalarOperator(
        mean,
        BroadcastedOperator(
            -,
            BroadcastedOperator(
                -,
                BroadcastedOperator(
                    xlogy,
                    Variable(y_unwraped),
                    BroadcastedOperator(
                        +,
                        Variable(x_unwraped),
                        Constant(eps(Float32))
                    ),
                ),
            ),
            BroadcastedOperator(
                xlogy,
                BroadcastedOperator(
                    -,
                    Constant(1),
                    Variable(y_unwraped)
                ),
                BroadcastedOperator(
                    +,
                    BroadcastedOperator(
                        -,
                        Constant(1),
                        Variable(x_unwraped)
                    ),
                    Constant(eps(Float32))
                ),
            ),
        ),
    )
    traverse_order = topological_sort(expr)
    @test map(node -> typeof(node), traverse_order) == [
        Variable{Vector{Float64}},
        Variable{Vector{Float32}},
        Constant{Float32},
        BroadcastedOperator{typeof(+),Vector{Float32}},
        BroadcastedOperator{typeof(xlogy),Vector{Float64}},
        BroadcastedOperator{typeof(-),Vector{Float64}},
        Constant{Int64},
        BroadcastedOperator{typeof(-),Vector{Float64}},
        BroadcastedOperator{typeof(-),Vector{Float32}},
        BroadcastedOperator{typeof(+),Vector{Float32}},
        BroadcastedOperator{typeof(xlogy),Vector{Float64}},
        BroadcastedOperator{typeof(-),Vector{Float64}},
        ScalarOperator{typeof(mean),Float64}
    ]
    @test expr.output == expected_crossentropy_evaluation

    flux_expr = Flux.Losses.binarycrossentropy(x, y)
    @test flux_expr.output == expected_crossentropy_evaluation
end
