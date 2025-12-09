using Test
using MMProjection
using Random
using Statistics

# -------------------------------
# Helper functions
# -------------------------------

function KKTCondition(λ, y, w, r)
    s = 0.0
    n = length(y)
    @inbounds @simd for i in 1:n
        yi = y[i]
        wi = w[i]
        diff = min(wi*(λ*wi - yi), 0)
        if diff < 0
            s += diff
        end
    end
    return s + r
end

# -------------------------------
# Unit Tests
# -------------------------------


@testset "Reservoir Sampling Tests" begin
    Random.seed!(1234)
    K = 100
    n = 1_000_000
    Sample = Int[]
    reservoir_indices!(Sample, K, n)

    @test length(Sample) == K
    @test 1 <= minimum(Sample) < maximum(Sample) <= n
    @test length(unique(Sample)) == K
end


@testset "Simplex MM Projection Tests" begin
    Random.seed!(1234)
    K, n = 100_000, 10_000_000
    Sample = Int[]
    reservoir_indices!(Sample, K, n)
    sort!(Sample);
    r = 50.0
    y = rand(n)
    w = rand(n)

    x, λ, iters = Simplex_MM_Projection(y, w, r; tol=1e-8, sample=Sample)

    # KKT check
    @test abs(KKTCondition(λ, y, w, r)) < 1e-4
end


@testset "Condat Projection Tests" begin
    Random.seed!(1234)
    n = 20000
    r = 10.0
    y = rand(n)
    w = rand(n)

    x, λ, iters = wcondat_p(y, w, r, Threads.nthreads())

    # KKT check
    @test abs(KKTCondition(λ, y, w, r)) < 1e-4
end

println("All MMProjection tests passed.")
