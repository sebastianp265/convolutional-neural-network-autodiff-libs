using Test: Random
using JLD2
using Statistics
using ProfileView

@testset "KM3" begin
    X_train = load("../data/KM3/imdb_dataset_prepared.jld2", "X_train")[:, 1:256]
    y_train = load("../data/KM3/imdb_dataset_prepared.jld2", "y_train")[:, 1:256]
    X_test = load("../data/KM3/imdb_dataset_prepared.jld2", "X_test")
    y_test = load("../data/KM3/imdb_dataset_prepared.jld2", "y_test")
    embeddings = load("../data/KM3/imdb_dataset_prepared.jld2", "embeddings")
    vocab = load("../data/KM3/imdb_dataset_prepared.jld2", "vocab")
    dataset = DataLoader((X_train, y_train), batchsize=64, shuffle=true)
    embedding_dim = size(embeddings, 1)
    @test size(embeddings) == (50, 12849)

    fixed_randn32(dims...) = randn32(Random.MersenneTwister(123), dims...)
    fixed_glorot(dims...) = glorot_uniform(Random.MersenneTwister(123), dims...)

    @test length(vocab) == 12849
    @test embedding_dim == 50

    flux_model = Flux.Chain(
        Flux.Embedding(length(vocab), embedding_dim, init=fixed_randn32),
        x -> permutedims(x, (2, 1, 3)),
        # Flux.Conv((3,), embedding_dim => 8, Flux.relu, init=fixed_glorot),
        Flux.MaxPool((8,)),
        Flux.flatten,
        Flux.Dense(800, 1, Flux.sigmoid, init=fixed_glorot)
    )
    my_model = Chain(
        Embedding(length(vocab), embedding_dim, init=fixed_randn32),
        x -> permutedims(x, (2, 1, 3)),
        # Conv((3,), embedding_dim => 8, relu, init=fixed_glorot),
        MaxPool((8,)),
        flatten,
        Dense(800, 1, sigmoid, init=fixed_glorot)
    )

    # forward pass
    t = @elapsed begin
        flux_output = flux_model(X_train)
        my_output = my_model(X_train).output
    end
    println("Flux_output $t seconds")
    t = @elapsed begin
        my_output = my_model(X_train).output
    end
    println("My_output $t seconds")
    @test size(flux_output) == size(my_output)
    @test flux_output == my_output

    # gradient pass
    t = @elapsed begin
        flux_grad = Flux.gradient(flux_model) do m 
        sum(m(X_train))
        end
    end
    println("Flux_gradient $t seconds")
    t = @elapsed begin
        my_grad = gradient!(my_model) do m
            sum(m(X_train))
        end
    end
    println("My_gradient $t seconds")
    @test flux_grad == my_grad
end
