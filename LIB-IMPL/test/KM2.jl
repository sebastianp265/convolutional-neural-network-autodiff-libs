using JLD2, Statistics
import Flux, Random

@testset "KM2" begin
    X_train = load("../data/imdb_dataset_prepared.jld2", "X_train")
    y_train = load("../data/imdb_dataset_prepared.jld2", "y_train")
    X_test = load("../data/imdb_dataset_prepared.jld2", "X_test")
    y_test = load("../data/imdb_dataset_prepared.jld2", "y_test")

    flux_dataset = Flux.DataLoader((X_train, y_train), batchsize=64, shuffle=false)
    my_dataset = DataLoader((X_train, y_train), batchsize=64, shuffle=false)

    flux_init(dims::Integer...) = begin
        Flux.glorot_uniform(Random.MersenneTwister(123), dims...)
    end
    flux_model = Flux.Chain(
        Flux.Dense(size(X_train, 1), 32, Flux.relu; init=flux_init),
        Flux.Dense(32, 1, Flux.sigmoid, init=flux_init)
    )

    my_init(dims::Integer...) = begin
        Flux.glorot_uniform(Random.MersenneTwister(123), dims...)
    end
    my_model = Chain(
        Dense(size(X_train, 1), 32, relu; init=my_init),
        Dense(32, 1, sigmoid, init=my_init)
    )

    @test flux_model == my_model

    flux_loss(m, x, y) = Flux.Losses.binarycrossentropy(m(x), y)
    my_loss(m, x, y) = binarycrossentropy(m(x), y)

    flux_accuracy(m, x, y) = mean((m(x) .> 0.5) .== (y .> 0.5))
    my_accuracy(m, x, y) = mean((m(x).output .> 0.5) .== (y .> 0.5))

    flux_opt = Flux.setup(Flux.Adam(), flux_model)
    my_opt = setup(Adam(), my_model)

    @test flux_opt == my_opt

    epochs = 5
    for epoch in 1:epochs
        flux_total_loss = 0.0
        flux_total_acc = 0.0

        my_total_loss = 0.0
        my_total_acc = 0.0

        num_samples = 0
        t = @elapsed begin
            for (flux_data, my_data) in zip(flux_dataset, my_dataset)
                flux_x, flux_y = flux_data
                my_x, my_y = my_data
                @test flux_x == my_x
                @test flux_y == my_y

                flux_grads = Flux.gradient(flux_model) do m
                    l = flux_loss(m, flux_x, flux_y)
                    flux_total_loss += l
                    flux_total_acc += flux_accuracy(m, flux_x, flux_y)
                    return l
                end
                my_grads = gradient!(my_model) do m
                    l = my_loss(m, my_x, my_y)
                    my_total_loss += l.output
                    my_total_acc += my_accuracy(m, my_x, my_y)
                    return l
                end

                @test flux_grads == my_grads

                Flux.Optimisers.update!(flux_opt, flux_model, flux_grads[1])
                update!(my_opt, my_model, my_grads[1])
                @test flux_opt == my_opt
                @test flux_model == my_model

                num_samples += 1
            end

            flux_train_loss = flux_total_loss / num_samples
            my_train_loss = my_total_loss / num_samples
            @test flux_train_loss == my_train_loss

            flux_train_acc = flux_total_acc / num_samples
            my_train_acc = my_total_acc / num_samples
            @test flux_train_acc == my_train_acc

            flux_test_acc = flux_accuracy(flux_model, X_test, y_test)
            my_test_acc = my_accuracy(my_model, X_test, y_test)
            @test flux_test_acc == my_test_acc

            flux_test_loss = flux_loss(flux_model, X_test, y_test)
            my_test_loss = my_loss(my_model, X_test, y_test).output
            @test flux_test_loss == my_test_loss
        end
    end

end
