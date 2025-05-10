import Flux
using Random

@testset "glorot_uniform works the same as in flux" begin
    flux_rng = MersenneTwister(42)
    my_rng = MersenneTwister(42)
    @test Flux.glorot_uniform(flux_rng, 3, 2) == glorot_uniform(my_rng, 3, 2)
end
