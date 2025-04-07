using Test
using MYMLP

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
