using Test: Random
using JLD2

@testset "KM3" begin
    X_train = load("../data/KM3/imdb_dataset_prepared.jld2", "X_train")
    y_train = load("../data/KM3/imdb_dataset_prepared.jld2", "y_train")
    X_test = load("../data/KM3/imdb_dataset_prepared.jld2", "X_test")
    y_test = load("../data/KM3/imdb_dataset_prepared.jld2", "y_test")
    embeddings = load("../data/KM3/imdb_dataset_prepared.jld2", "embeddings")
    vocab = load("../data/KM3/imdb_dataset_prepared.jld2", "vocab")

    embedding_dim = size(embeddings, 1)
    @test size(embeddings) == (50, 12849)

    rng(dims...) = randn32(Random.MersenneTwister(123), dims...)

    @test length(vocab) == 12849
    @test embedding_dim == 50

    @test size(X_train) == (130, 40000)
    flux_model = Flux.Chain(
        Flux.Embedding(length(vocab), embedding_dim, init=rng), # output = (50, 130, 40000)
        x -> permutedims(x, (2, 1, 3)) # output (130, 50, 40000)
    )
    my_model = Chain(
        Embedding(length(vocab), embedding_dim, init=rng),
        x -> permutedims(x, (2, 1, 3)),
    )

    @test flux_model(X_train) == my_model(X_train)
end
