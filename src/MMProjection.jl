module MMProjection


include("UtilityFunctions.jl")
include("HalfspaceFunctions.jl") 
include("DykstraFuntions.jl")  
include("BallFunctions.jl") 
include("CovarianceMatrixShrinkage.jl") 
include("MeanVariancePortforlia.jl")
include("OptimalTransportSolvers.jl") 
include("WeightedSimplex.jl") 

using .UtilityFunctions
using .HalfspaceFunctions
using .DykstraFuntions
using .BallFunctions
using .CovarianceMatrixShrinkage
using .MeanVariancePortforlia
using .OptimalTransportSolvers
using .WeightedSimplex

export project_with_balls_or_spheres
export radii_from_target
export cov_shrinkage_newton, nearest_cov_shrinkage_dykstra_AAI, nearest_cov_shrinkage_dykstra, corr_metrics
export run_dykstra!, anderson_I_state!, anderson_II_state!
export project_halfspaces_active_set
export mean_variance_mle_gradient
export solve_mm_ot
export make_set, build_sets, Loss, prox, make_projector, IndWSimplex, pav_nondecreasing!, pav_nonincreasing!
export CenteredBallL2, CenteredSphereL2
export Simplex_MM_Projection, wcondat_p, reservoir_indices!

const DEFAULT_SAMPLE = let
    K, n = 100_000, 10_000_000
    S = Int[]
    reservoir_indices!(S, K, n)
    sort!(S)
    S
end

end # module MMProjection
