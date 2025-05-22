using Test: Random
using JLD2
using Printf, Statistics
using ProfileView


@testset "KM3" begin
    X_train = load("../data/KM3/imdb_dataset_prepared.jld2", "X_train")[:,1:256]
    y_train = load("../data/KM3/imdb_dataset_prepared.jld2", "y_train")[:,1:256]
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

    @test size(X_train) == (40000, 64)

    model = Flux.Chain(
        Flux.Embedding(length(vocab), embedding_dim, init=fixed_randn32),
        x -> permutedims(x, (2, 1, 3)),
        Flux.Conv((3,), embedding_dim => 8, Flux.relu, init=fixed_glorot),
        Flux.MaxPool((8,)),
        Flux.flatten,
        Flux.Dense(128, 1, Flux.sigmoid)
    )
    flux_model = model
    my_model = Chain(
        Embedding(length(vocab), embedding_dim, init=fixed_randn32),
        x -> permutedims(x, (2, 1, 3)),
        Conv((3,), embedding_dim => 8, relu, init=fixed_glorot),
        MaxPool((8,)),
        flatten,
        Dense(128, 1, sigmoid)
    )

    test = Chain(
        Embedding(length(vocab), embedding_dim, init=fixed_randn32),
        x -> permutedims(x, (2, 1, 3)),
        MaxPool((8,)),
        #flatten,
        Dense(128, 1, relu)
    )

    
    my_model.layers[1].weight.output .= embeddings

    #@test size(flux_conv.weight) == size(my_conv.weight.output)
    #@test flux_conv.weight == my_conv.weight.output
    #@test flux_conv.bias == my_conv.bias.output
    #@test flux_conv.stride == my_conv.stride
    #@test flux_conv.pad == my_conv.pad
    #@test flux_conv.dilation == my_conv.dilation
    #@test flux_conv.groups == my_conv.groups

    #@test flux_model(X_train) == my_model(X_train).output
    #flux_grad = Flux.gradient(flux_model) do m 
    #    sum(m(X_train))
    #end
    #@test size(flux_model(X_train)) == size(my_model(X_train))
    #@test flux_model(X_train) == my_model(X_train).output
    #@show flux_model(X_train)
    #t = @elapsed begin
    #my_grad = gradient!(my_model) do m 
    #   sum(m(X_train))
    #end
    #end 
    #println("My_model", t)
    #t = @elapsed begin
    #flux_grad = Flux.gradient(flux_model) do m 
    #    sum(m(X_train))
    #end
    #end 
    #println("Flux", t)
    
    
    #@show my_grad
    #@show flux_grad
    #@show my_grad
    #@test flux_grad == my_grad
    

    loss(m, x, y) = binarycrossentropy(m(x), y)
    accuracy(m, x, y) =  mean((m(x).output .> 0.5) .== (y .> 0.5))

    i = 0
    opt = setup(Adam(), my_model)
    epochs = 2
    @show i

        for epoch in 1:epochs
            total_loss = 0.0
            total_acc = 0.0
            num_samples = 0

            t = @elapsed begin
                for (x, y) in dataset
                    i= i + 1
                    grads = gradient!(my_model) do m
                        l = loss(m, x, y)
                        #total_loss += l.output
                        #total_acc += accuracy(m, x, y)
                        return l
                    end
                    update!(opt, my_model, grads[1])
                    num_samples += 1
                end

                #train_loss = total_loss / num_samples
                #train_acc = total_acc / num_samples

                #test_acc = accuracy(my_model, X_test, y_test)
                #test_loss = loss(my_model, X_test, y_test).output
            end

            #println(@sprintf("Epoch: %d (%.2fs) \tTrain: (l: %.2f, a: %.2f) \tTest: (l: %.2f, a: %.2f)", 
            #    epoch, t, train_loss, train_acc, test_loss, test_acc))
            println(t)
            #println(train_acc)
        end
    #ProfileView.@profview sum(1:5)
    #@profview profile_test()

    

end
