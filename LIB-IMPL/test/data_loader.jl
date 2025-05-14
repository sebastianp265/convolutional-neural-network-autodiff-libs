function test_dataset_impl(dataloader, x, y, expected; batch_size=2)
    dataset = dataloader((x, y), batchsize=batch_size, shuffle=false)
    batch_index = 1

    for (x_batch, y_batch) in dataset
        @testset "Batch $batch_index" begin
            @test x_batch == expected[1][batch_index]
            @test y_batch == expected[2][batch_index]
        end
        batch_index += 1
    end
end

@testset "DataLoader Iteration Without Shuffling" begin
    x = Float32[
        1.0 2.0 3.0 4.0 5.0;
        5.0 6.0 7.0 8.0 9.0;
        9.0 10.0 11.0 12.0 13.0;
        13.0 14.0 15.0 16.0 17.0
    ] # (4 features, 5 samples)
    y = Float32[
        100.0 200.0 300.0 400.0 500.0
    ] # (1 feature, 5 samples)

    expected = (
        (x[:, 1:2], x[:, 3:4], x[:, 5:5]),
        (y[:, 1:2], y[:, 3:4], y[:, 5:5])
    )
    test_dataset_impl(
        Flux.DataLoader,
        x,
        y,
        expected
    )
    test_dataset_impl(
        DataLoader,
        x,
        y,
        expected
    )
end
