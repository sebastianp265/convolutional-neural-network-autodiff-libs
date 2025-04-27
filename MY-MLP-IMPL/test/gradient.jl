function test_flux_gradient(expected, f::Function, args...)
    println("Measuring time for Flux.gradient...")
    flux_time = @elapsed begin
        @test Flux.gradient(f, args...) == expected
    end
    println("Flux.gradient time: $(flux_time*1000) ms")
end

function test_my_gradient(expected, f::Function, args...)
    println("Measuring time for custom gradient!...")
    my_time = @elapsed begin
        @test gradient!(f, args...) == expected
    end
    println("Custom gradient! time: $(my_time*1000) ms")
end

@testset "Computing Gradient" begin
    x1 = [0, π, 2π]
    x2 = [1, 2, 4]

    expected = (cos.(x1) .+ x2, x1)
    f(x1, x2) = sum(x1 .* x2 .+ sin.(x1))

    test_flux_gradient(expected, f, x1, x2)

    my_x1 = Variable(x1)
    my_x2 = Variable(x2)
    test_my_gradient(expected, f, my_x1, my_x2)
end

@testset "Computing Gradient in batches" begin
    x1 = [
        0 0;
        π 2π;
        2π 4π
    ]
    x2 = [
        1 2;
        2 4;
        4 8
    ]

    expected = (cos.(x1) .+ x2, x1)
    f(x1, x2) = sum(x1 .* x2 .+ sin.(x1))

    test_flux_gradient(expected, f, x1, x2)

    my_x1 = Variable(x1)
    my_x2 = Variable(x2)
    test_my_gradient(expected, f, my_x1, my_x2)
end

@testset "Manual Gradient Computation on MLP" begin
    x = Float32[
        0.5 0.2;
        0.1 0.4;
        0.3 0.8
    ] # shape (3, 2)
    y = Float32[1.0 0.0] # shape(1, 2)

    @assert size(x) == (3, 2)
    @assert size(y) == (1, 2)

    c = Flux.Chain(
        Flux.Dense(3, 1, Flux.sigmoid)
    )

    c[1].weight .= Float32[0.5 0.1 0.1]
    c[1].bias .= Float32[0.1]

    expected = (
        (
        layers=(
            (
            weight=Float32[-0.04299689 0.095678955 0.17117205],
            bias=Float32[0.08780344],
            σ=nothing
        ),
        ),
    ),
    )
    flux_loss(c) = Flux.Losses.binarycrossentropy(c(x), y)
    test_flux_gradient(
        expected,
        flux_loss,
        c
    )

    my_c = Chain(
        Dense(3, 1, sigmoid)
    )

    my_c.layers[1].weight.output .= Float32[0.5 0.1 0.1]
    my_c.layers[1].bias.output .= Float32[0.1]

    my_loss(c) = binarycrossentropy(c(x), y)
    test_my_gradient(
        expected,
        my_loss,
        my_c
    )
end
