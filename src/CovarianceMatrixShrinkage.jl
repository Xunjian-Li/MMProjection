module CovarianceMatrixShrinkage

using LinearAlgebra
using ForwardDiff

include("DykstraFuntions.jl")
using .DykstraFuntions

export cov_shrinkage_newton, nearest_cov_shrinkage_dykstra_AAI, nearest_cov_shrinkage_dykstra, corr_metrics

"""
Compute metrics for projected correlation matrix:
  - Frobenius mismatch vs true projection
  - Minimum eigenvalue (PSD check)
"""
function corr_metrics(X::Matrix)
    # diag = 1 is guaranteed by algorithm
    # check PSD
    λmin = minimum(eigvals(Symmetric(X)))
    # check deviation from last PSD projection
    Fval = sum((X .- proj_psd!(copy(X))).^2)
    return Fval, λmin
end

# -----------------------------
# Covariance shrinkage via Newton in λ
# -----------------------------
"""
    cov_shrinkage_newton(S, T, r;
                         λ0=0.0, tol=1e-6, maxiter=50)

Solve

    minimize  1/2 ‖Σ - S‖_F^2
    subject to Σ ⪰ 0,  ‖Σ - T‖_F ≤ r,

where `S` is the sample covariance, `T` is the shrinkage target,
and `r` is the radius of the Frobenius ball around `T`.

Returns
-------
λ̂    : optimal dual scalar λ ≥ 0
Σ̂    : shrunk covariance matrix Σ(λ̂)
iters : Newton iterations used
hist  : vector of (g, |g|) where g = D'(λ)
"""
function cov_shrinkage_newton(
    S::AbstractMatrix,
    T::AbstractMatrix,
    r::Real;
    λ0::Real = 0.0,
    tol::Real = 1e-6,
    maxiter::Int = 50,
)

    @assert size(S) == size(T) "S, T must have same size"
    @assert issymmetric(S) && issymmetric(T) "S, T should be symmetric"

    λ = max(λ0, 0.0)

    function Dprime(λ::Real)
        α = 1.0 / (1.0 + λ)
        Cλ = α .* (S .+ λ .* T)            # (S + λT)/(1+λ)
        Σλ = proj_psd!(Cλ)
        # D'(λ) = 1/2 ( ||Σ(λ)-T||_F^2 - r^2 )
        return 0.5 * (sum(abs2, Σλ .- T) - r^2), Σλ
    end

    hist = Vector{Tuple{Float64,Float64}}()
    Σλ = similar(S)

    iters = maxiter
    for k in 1:maxiter
        g, Σλ = Dprime(λ)
        push!(hist, (g, abs(g)))

        if abs(g) < tol
            iters = k
            break
        end

        φ(λ_) = first(Dprime(λ_))
        h = ForwardDiff.derivative(φ, λ)

        if abs(h) < 1e-12
            iters = k
            break
        end

        step = g / h
        λ_new = λ - step

        if λ_new < 0
            λ_new = 0.5 * λ
        end

        if abs(λ_new - λ) < tol * (1 + abs(λ))
            λ = λ_new
            iters = k
            g, Σλ = Dprime(λ)
            break
        end

        λ = λ_new
    end

    λ̂ = max(λ, 0.0)
    Σ̂ = Σλ

    return λ̂, Σ̂, iters, hist
end


"""
    proj_psd!(C)

Project a symmetric matrix `C` onto the PSD cone:

    P₊(C) = Q * max.(Λ, 0) * Qᵀ,

where C = Q * Diagonal(Λ) * Qᵀ.
The projection is done *in-place*.
"""
function proj_psd!(C::AbstractMatrix)
    F = eigen(Symmetric(C))
    λplus = max.(F.values, zero(eltype(F.values)))
    C .= F.vectors * Diagonal(λplus) * F.vectors'
    return C
end


"""
    proj_frob_ball!(X, T, r)

Project `X` onto the Frobenius ball

    B = { Σ : ‖Σ - T‖_F ≤ r }.

`T` is the center, `r` is the radius (> 0).
Projection is done *in-place* on `X`.

If X is already inside the ball, it is left unchanged.
"""
function proj_frob_ball!(X::AbstractMatrix{T},
                         Tcenter::AbstractMatrix{T},
                         r::Real) where {T<:Real}
    @assert size(X) == size(Tcenter) "X and Tcenter must have the same size"

    Δ = X .- Tcenter
    nrm = norm(Δ)                         # Frobenius norm for matrices

    if nrm ≤ r || nrm == 0
        # Already inside the ball (or exactly at the center)
        return X
    else
        α = T(r) / nrm                    # shrink factor, cast r to element type T
        @. X = Tcenter + α * Δ           # X ← T + (r/‖Δ‖) * (X - T)
        return X
    end
end


# ============================================================
# 2. Plain Dykstra algorithm: PSD ∩ Frobenius ball
# ============================================================

"""
    nearest_cov_shrinkage_dykstra(S, T, r;
                                  maxiter=5000, tol=1e-8, verbose=false)

Solve the covariance shrinkage projection

    min  1/2 ‖Σ - S‖_F^2
    s.t. Σ ∈ C₁ ∩ C₂,
         C₁ = { Σ : Σ ⪰ 0 },
         C₂ = { Σ : ‖Σ - T‖_F ≤ r },

via Dykstra's algorithm with:

    C₁ = PSD cone,
    C₂ = Frobenius ball centred at T with radius r.

Returns

    Σ̂      – projected matrix (as Symmetric)
    iters  – number of Dykstra iterations
    ok     – convergence flag (true if rel. change < tol)
"""
function nearest_cov_shrinkage_dykstra(
    S::AbstractMatrix{T},
    Tcenter::AbstractMatrix{T},
    r::Real;
    maxiter::Int = 5_000,
    tol::Real    = 1e-8,
    verbose::Bool = false
) where {T<:Real}

    @assert size(S,1) == size(S,2)       "S must be square"
    @assert size(S) == size(Tcenter)     "S and Tcenter must have same size"
    @assert issymmetric(S)               "S must be symmetric"

    n = size(S,1)

    # Current iterate
    X  = Matrix(S)

    # Dykstra correction terms for the two sets
    P1 = zeros(T, n, n)    # for PSD
    P2 = zeros(T, n, n)    # for Frobenius ball

    Y = similar(X)
    Z = similar(X)

    ok    = false
    iters = maxiter

    for k in 1:maxiter
        X_old = copy(X)

        # ---- Step 1: project onto PSD cone ----
        @. Y = X + P1
        copyto!(X, Y)
        proj_psd!(X)
        @. P1 = Y - X

        # ---- Step 2: project onto Frobenius ball ----
        @. Z = X + P2
        copyto!(X, Z)
        proj_frob_ball!(X, Tcenter, r)
        @. P2 = Z - X

        # ---- Relative change stopping criterion ----
        relchg = norm(X .- X_old) / max(one(T), norm(X_old))
        if verbose
            @info "Dykstra iter = $k   relchg = $relchg"
        end

        if relchg < tol
            ok    = true
            iters = k
            break
        end
    end

    return Symmetric(X), iters, ok
end


# ============================================================
# 3. Dykstra state map G(s) for Anderson acceleration
# ============================================================

"""
    make_G_dykstra(Tcenter, r)

Construct a state-transition map G(s) that performs **one full
Dykstra iteration** for the shrinkage problem.

The state vector is

    s = [ vec(X);
          vec(P1);
          vec(P2) ] ∈ ℝ^{3n²},

where

    X  – current iterate
    P1 – Dykstra correction for PSD constraint
    P2 – Dykstra correction for Frobenius-ball constraint

The returned function `G_state(s)` performs

    (X, P1, P2) ↦ (X_new, P1_new, P2_new)

and returns the new state vector with the same layout.
`Tcenter` and `r` are captured by closure.
"""
function make_G_dykstra(Tcenter::AbstractMatrix{T},
                        r::Real) where {T<:Real}
    n = size(Tcenter,1)
    N = n*n

    function G_state(s::AbstractVector{T})
        @assert length(s) == 3N "State vector must have length 3n²."

        # ---- Unpack state ----
        X  = reshape(view(s, 1:N),        n, n)
        P1 = reshape(view(s, N+1:2N),     n, n)
        P2 = reshape(view(s, 2N+1:3N),    n, n)

        # ---- Step 1: PSD projection ----
        Y = X .+ P1
        Z = copy(Y)
        proj_psd!(Z)
        P1_new = Y .- Z

        # ---- Step 2: Frobenius-ball projection ----
        W = Z .+ P2
        X_new = copy(W)
        proj_frob_ball!(X_new, Tcenter, r)
        P2_new = W .- X_new

        # ---- Repack state ----
        s_new = similar(s)
        s_new[1:N]           .= vec(X_new)
        s_new[N+1:2N]        .= vec(P1_new)
        s_new[2N+1:3N]       .= vec(P2_new)

        return s_new
    end

    return G_state
end


# ============================================================
# 4. Dykstra + Anderson Acceleration (Type I)
# ============================================================

# Here we assume you already have an implementation of
# `anderson_I_state!(G, s0; m, maxiter, tol, τ, λ, cnd_max)`
# from your previous experiments. We simply wrap it.

"""
    nearest_cov_shrinkage_dykstra_AAI(S, T, r;
                                      m=8, maxiter=1000,
                                      tol=1e-6, τ=1.0,
                                      λ=1e-12, cnd_max=1e6)

Nearest covariance shrinkage projection using:

  • Dykstra fixed-point iteration for
        C₁ = PSD cone, C₂ = Frobenius ball,
  • Type-I Anderson acceleration (Walker–Ni) applied
    to the Dykstra state map.

Arguments
---------
- `S`       : sample covariance matrix (symmetric)
- `T`       : shrinkage target matrix (same size as S)
- `r`       : Frobenius-ball radius
- `m`       : Anderson memory parameter
- `maxiter` : maximum Anderson iterations
- `tol`     : stopping tolerance on fixed-point residual
- `τ, λ, cnd_max` : AA tuning parameters

Returns
-------
- `Σ̂`   : projected matrix (Symmetric)
- `iters`: number of outer AA iterations
- `ok`   : convergence flag from AA routine
"""
function nearest_cov_shrinkage_dykstra_AAI(
    S::AbstractMatrix{T},
    Tcenter::AbstractMatrix{T},
    r::Real;
    m::Int = 8,
    maxiter::Int = 1_000,
    tol::Real = 1e-6,
    τ::Real = 1.0,
    λ::Real = 1e-12,
    cnd_max::Real = 1e6
) where {T<:Real}

    @assert size(S) == size(Tcenter)
    @assert issymmetric(S)

    n  = size(S,1)
    N  = n*n

    # Build fixed-point map G for the shrinkage Dykstra
    G = make_G_dykstra(Tcenter, r)

    # Initial state: X = S, P1 = 0, P2 = 0
    s0 = zeros(T, 3N)
    s0[1:N] .= vec(S)

    # Use your existing Type-I AA routine
    s_star, iters, ok =
        anderson_I_state!(G, s0;
                          m=m, maxiter=maxiter,
                          tol=tol, τ=τ,
                          λ=λ, cnd_max=cnd_max)

    X_star = reshape(view(s_star, 1:N), n, n)
    return Symmetric(X_star), iters, ok
end


end

