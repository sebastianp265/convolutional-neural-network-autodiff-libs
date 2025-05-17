using Test: Random
using JLD2

@testset "KM3" begin
    X_train = load("../data/KM3/imdb_dataset_prepared.jld2", "X_train")[:, 1:64]
    embeddings = load("../data/KM3/imdb_dataset_prepared.jld2", "embeddings")
    vocab = load("../data/KM3/imdb_dataset_prepared.jld2", "vocab")

    embedding_dim = size(embeddings, 1)
    @test size(embeddings) == (50, 12849)

    fixed_randn32(dims...) = randn32(Random.MersenneTwister(123), dims...)
    fixed_glorot(dims...) = glorot_uniform(Random.MersenneTwister(123), dims...)

    @test length(vocab) == 12849
    @test embedding_dim == 50

    @test size(X_train) == (130, 64)

    flux_model = Flux.Chain(
        Flux.Embedding(length(vocab), embedding_dim, init=fixed_randn32),
        x -> permutedims(x, (2, 1, 3)),
        Flux.Conv((3,), embedding_dim => 8, Flux.relu, init=fixed_glorot),
        Flux.flatten,
    )
    my_model = Chain(
        Embedding(length(vocab), embedding_dim, init=fixed_randn32),
        x -> permutedims(x, (2, 1, 3)),
        Conv((3,), embedding_dim => 8, relu, init=fixed_glorot),
        flatten,
    )

    flux_conv = flux_model[3]
    my_conv = my_model.layers[3]
    
    @test size(flux_conv.weight) == size(my_conv.weight.output)
    @test flux_conv.weight == my_conv.weight.output
    @test flux_conv.bias == my_conv.bias.output
    @test flux_conv.stride == my_conv.stride
    @test flux_conv.pad == my_conv.pad
    @test flux_conv.dilation == my_conv.dilation
    @test flux_conv.groups == my_conv.groups

    @test flux_model(X_train) == my_model(X_train).output
    flux_grad = Flux.gradient(flux_model) do m 
        sum(m(X_train))
    end
    my_grad = gradient!(my_model) do m 
        sum(m(X_train))
    end
    @test flux_grad == my_grad
end
