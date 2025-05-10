@testset "Adam optimizer initialization" begin
    flux_model = Flux.Chain(
        Flux.Dense(3, 1, Flux.sigmoid)
    )
    flux_model[1].weight .= Float32[0.5 0.1 0.1]
    flux_model[1].bias .= Float32[0.1]

    my_model = Chain(
        Dense(3, 1, sigmoid)
    )
    my_model.layers[1].weight.output .= Float32[0.5 0.1 0.1]
    my_model.layers[1].bias.output .= Float32[0.1]

    @test flux_model == my_model


    flux_opt = Flux.setup(Flux.Adam(), flux_model)
    my_opt = setup(Adam(), my_model)

    @test flux_opt == my_opt

    grad = (
        layers=(
            (weight=Float32[-0.04 0.10 0.17],
            bias=Float32[0.09],
            σ=nothing),
        ),
    )

    Flux.Optimisers.update!(flux_opt, flux_model, grad)
    update!(my_opt, my_model, grad)

    @test flux_opt == my_opt
    @test flux_model == my_model
    grad = (
        layers=(
            (weight=Float32[-0.02 -0.5 0.1],
            bias=Float32[0.04],
            σ=nothing),
        ),
    )

    Flux.Optimisers.update!(flux_opt, flux_model, grad)
    update!(my_opt, my_model, grad)

    @test flux_opt == my_opt
    @test flux_model == my_model
end

