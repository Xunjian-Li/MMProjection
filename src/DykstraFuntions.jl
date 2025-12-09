########################
# Dykstra + Anderson Acceleration (Type-I / Type-II)
# Minimal, formula-focused English comments
########################

module DykstraFuntions

include("UtilityFunctions.jl") 
using .UtilityFunctions

using LinearAlgebra  # eigvals, Symmetric
using ProximalOperators

export run_dykstra!, anderson_I_state!, anderson_II_state!

# ---- Projector wrapper: write P_{C_i}(v) into `dst`
default_project!(dst, Ci, v) = (dst .= first(prox(Ci, v)); dst)

# ---- Finite-check
@inline _isfinitevec(v) = all(isfinite, v)

# ---- Classic (sequential) Dykstra map G(state) = [x_{k+1}; vec(P_{k+1})]
# State layout: s = [x; vec(P)] ∈ ℝ^{p + p r}, with P = [p_1 … p_r].
# One sweep (k → k+1), for i = 1..r:
#   v_i   = z_{i-1} - p_i          (z_0 := x_k)
#   w_i   = P_{C_i}(v_i)
#   p_i^+ = w_i - v_i
#   z_i   = w_i
# Result: x_{k+1} = z_r,  P_{k+1} = [p_1^+ … p_r^+]
function make_dykstra_sequential_map(y::AbstractVector{T}, sets;
                                     project!::Function = default_project!) where {T<:Real}
    p = length(y)
    r = length(sets)
    # Work buffers reused inside the closure (sequential ⇒ only 2 vecs needed)
    z     = similar(y)        # running point z_i
    vbuf  = similar(y)        # v_i
    wbuf  = similar(y)        # w_i = P_{C_i}(v_i)
    Pnext = zeros(T, p, r)    # next corrections [p_1^+ … p_r^+]

    function G(state::AbstractVector{T})
        @views xk = state[1:p]
        @views Pk = reshape(state[p+1:end], p, r)

        # z_0 = x_k
        z .= xk

        # sequential sweep over i = 1..r
        for i in 1:r
            @views pi  = Pk[:, i]
            @views pin = Pnext[:, i]

            @. vbuf = z - pi                # v_i = z_{i-1} - p_i
            project!(wbuf, sets[i], vbuf)   # w_i = P_{C_i}(v_i)
            @. pin = wbuf - vbuf            # p_i^+ = w_i - v_i
            z .= wbuf                       # z_i = w_i
        end
        
        # x_{k+1} = z_r
        return vcat(z, vec(Pnext))
    end
    return G, p, r
end

# ---- Solve (ΔR'ΔR + λ I) c = ΔR' r_k   [ridge-regularized normal equations]
# This is the LS for AA:  min_c ||ΔR c − r_k||_2^2  (+ ridge)
# Uses Gram matrix (mm×mm) and eigenvalues to check conditioning: stable & fast.
function solve_lsq_coeffs(ΔR::AbstractMatrix, rk::AbstractVector;
                          λ::Real = 1e-12, cnd_max::Real = 1e8)

    mm = size(ΔR, 2)
    if mm == 0
        return (similar(rk, 0), true)             # nothing to solve
    end

    # guard finite inputs
    if any(!isfinite, ΔR) || any(!isfinite, rk)
        return (zeros(eltype(rk), mm), true)
    end

    # Gram matrix G = ΔR'ΔR  (small: (mm×mm))
    G = Symmetric(ΔR' * ΔR)

    # guard finite Gram
    if any(!isfinite, G)
        return (zeros(eltype(rk), mm), true)
    end

    # cond2(ΔR) = sqrt(cond2(G)) via eigenvalues of G (symmetric, real)
    vals = eigvals(G)
    vmax = maximum(vals)
    vmin = minimum(vals)
    if !(isfinite(vmax) && isfinite(vmin)) || vmax <= 0
        return (zeros(eltype(rk), mm), true)
    end
    if vmin <= eps(real(vmax)) * 1e2            # treat tiny min-eig as singular
        return (zeros(eltype(rk), mm), true)
    end
    condΔR = sqrt(vmax / vmin)
    if condΔR > cnd_max
        return (zeros(eltype(rk), mm), true)
    end

    # ridge + solve
    @inbounds @simd for i in 1:mm
        G[i,i] += λ
    end
    rhs = ΔR' * rk
    c = try
        G \ rhs
    catch
        fill!(rhs, zero(eltype(rhs)))
        rhs
    end
    if any(!isfinite, c)
        return (c, true)
    end
    return (c, false)
end


# ============================================================
# Top-level driver: Dykstra + AA on the *state* s = [x; vec(Λ)]
# Choose variant = :AAI (stable default) or :AAII (more aggressive).
# Convergence metric only checks the x-part: max |x_{k+1} − x_k|.
# ============================================================
function run_dykstra!(
    x::AbstractVector{T},
    y::AbstractVector{T},
    Λ::AbstractMatrix{T},
    sets::AbstractVector;
    AAvariant::Symbol = :AAI,      # :AAI or :AAII
    m::Int = 6,
    tol::T = T(1e-6),
    maxiter::Int = 2_000,
    τ::T = T(1.0),
    λ::T = T(1e-12),
    cnd_max::T = T(1e8),
    project! = default_project!
) where {T<:Real}

    p = length(y); r = size(Λ,2)
    @assert size(Λ,1) == p
    @assert length(sets) == r

    # Build G for the Dykstra one-step map
    G, pG, rG = make_dykstra_sequential_map(y, sets; project!)
    
    @assert pG == p && rG == r
    
    # Initial state: x0 = y − Σ_i Λ[:,i]  (state carries λ, not rλ)
    x0 = copy(y)
    for i in 1:r
        @views @. x0 -= Λ[:, i]
    end
    state0 = vcat(x0, vec(Λ))
    
    # Run: AA-I / AA-II / Plain (no AA)
    s_star = state0
    iters  = 0
    ok     = false

    if AAvariant === :AAI
        s_star, iters, ok = anderson_I_state!(G, state0;
            m=m, maxiter=maxiter, tol=tol, τ=τ, λ=λ, cnd_max=cnd_max)

    elseif AAvariant === :AAII
        s_star, iters, ok = anderson_II_state!(G, state0;
            m=m, maxiter=maxiter, tol=tol, τ=τ, λ=λ, cnd_max=cnd_max)
        
    elseif AAvariant === :none
        s_star, iters, ok = dykstra_serial!(G, state0; 
            m=m, maxiter=maxiter, tol=tol, τ=τ, λ=λ, cnd_max=cnd_max)
    
    else
        error("AAvariant must be :AAI, :AAII, or :none")
    end
    
    # Unpack back to x, Λ
    @views x .= s_star[1:p]
    @views Λ .= reshape(s_star[p+1:end], p, r)

    return x, Λ, iters, ok
end

"""
Plain fixed-point iteration (no AA), same interface as AAI/AAII.

Update:
    s_{k+1} = (1 - τ) s_k + τ G(s_k)

Notes:
  - Keeps (m, λ, cnd_max) in the signature for drop-in compatibility.
  - Converges when `conv(s_k, f_k) ≤ tol`, where f_k = G(s_k).
  - Guards NaN/Inf to fail fast and return last finite iterate.

Args (kept for compatibility):
  m, λ, cnd_max : ignored here
  τ ∈ (0, 1]    : relaxation (τ=1 -> classic Dykstra/Picard)
  tol           : convergence threshold on conv(s, f)
  conv          : e.g., (s,f) -> maximum(abs.(f .- s))
"""
function dykstra_serial!(G::Function, s0::AbstractVector{T};
                              m::Int=8, maxiter::Int=1_000, tol::Real=1e-8,
                              τ::Real=1.0, λ::Real=1e-12, cnd_max::Real=1e6) where {T<:Real}

    to = TimerOutput()  # keep symmetry with AA variants

    n = length(s0)
    s = copy(s0)          # s_k
    f = G(s)              # f_k = G(s_k)
    r = f .- s            # residual (kept for symmetry / debugging)

    iters = maxiter
    for k in 1:maxiter
        iters = k

        # convergence check
        if norm(s.- f, Inf) ≤ tol
            return f, iters, true
        end

        # relaxed Picard step: s_{k+1} = (1-τ)s_k + τ f_k
        @. s = (one(T) - τ) * s + τ * f

        # next iterate and residual
        f = G(s)
        r .= f .- s

    end

    return f, iters, false
end


"""
Walker–Ni Anderson Acceleration (Type-II) with *adaptive memory shrinking*.

Update:
    s_{k+1}^AA = f_k − τ ΔF_k γ,
    γ solves (ΔR_k'ΔR_k + λI) γ = ΔR_k' r_k,  r_k = f_k − s_k.

Stability trick (your requested logic):
  - Start with mm = q−1 columns (q = #history used this step).
  - If cond(ΔR[:, 1:mm]) > cnd_max (or non-finite), decrement mm and retry.
  - If mm < 1, fall back to simple iteration s_{k+1} = f_k.

Args:
  m        : maximum memory (uses up to m columns of differences)
  τ        : damping in (0,1]; τ=1 is full AA step
  λ        : small ridge term for normal equations
  cnd_max  : condition-number threshold for ΔR submatrix
  tol      : convergence threshold on `conv(s_k, f_k)`
  conv     : convergence metric function, default sup-norm of (f − s)
"""

function anderson_II_state!(G::Function, s0::AbstractVector{T};
                           m::Int=8, maxiter::Int=1_000, tol::Real=1e-6,
                           τ::Real=1.0, λ::Real=1e-12, cnd_max::Real=1e6) where {T<:Real}

    to = TimerOutput()
    
    n = length(s0)
    s = copy(s0)               # current input s_k
    f = G(s)                   # simple iterate f_k = G(s_k)
    r = f .- s                 # residual r_k
    
    f_old = zeros(n)                   # simple iterate f_k = G(s_k)
    r_old = zeros(n)                   # residual r_k
    s_old = zeros(n)
    RtR1 = zeros(m,m)
    
    ind = []
    # Difference buffers (capacity m columns)
    ΔF = zeros(T, n, m)        # ΔF[:, j] = F[:, j] − F[:, j+1]
    ΔR = similar(ΔF)           # ΔR[:, j] = R[:, j] − R[:, j+1]
    ΔX = similar(ΔF)           # ΔX[:, j] = X[:, j] − X[:, j+1]
    rhs = similar(s)

    sAA = similar(s)
    iters = 0
    for k in 1:maxiter
        iters = k
        j = (k+m-2) % m + 1

        # Convergence check
        if norm(s.- f, Inf) ≤ tol
            return f, iters, true
        end

        # Default next input is the plain fixed-point step (no acceleration).
        sAA .= f

        if k >= 2
            
            if k <= m+1
                ind = append!(ind, k-1)
            else
                ind[j] = k-1
            end
            
            ind1 = sortperm(ind; rev=true)
            ΔR[:,j] = r  - r_old
            ΔF[:,j] = f  - f_old
                    
            @views ΔRsub = ΔR[:,ind1]
            @views ΔFsub = ΔF[:,ind1]
            
            mul!(RtR1, transpose(ΔR), ΔR)
                    
            RtR = RtR1[ind1,ind1]
                    
            @views RtR[diagind(RtR)] .+= λ

#             mul!(rhs, transpose(ΔXsub), r)
            rhs = ΔRsub' * r

            γ = try
                RtR \ rhs
            catch
                fill!(rhs, zero(eltype(rhs)))  # emergency fallback
                rhs
            end

            # Anderson update: s_{k+1}^AA = f_k − τ ΔF γ
            copyto!(sAA, f)
            mul!(sAA, ΔFsub, γ, -τ, one(T))            # sAA = 1*sAA + (-τ)*(Fsub*γ) 

        end

        # Advance to next iteration: recompute f_{k+1}, r_{k+1}
        f_old .= f
        r_old .= r
        s_old .= s
        s .= sAA   ## x_{i+1}
        f = G(s)     ## f(x_{i+1})
        r .= f .- s  ## is f(x_{i+1}) - x_{i+1}
    end
    
    return f, iters, false
end


function anderson_I_state!(G::Function, s0::AbstractVector{T};
                           m::Int=8, maxiter::Int=1_000, tol::Real=1e-6,
                           τ::Real=1.0, λ::Real=1e-12, cnd_max::Real=1e6) where {T<:Real}
    
    n = length(s0)
    s = copy(s0)               # current input s_k
    f = G(s)                   # simple iterate f_k = G(s_k)
    r = f .- s                 # residual r_k
        
    f_old = zeros(n)                   # simple iterate f_k = G(s_k)
    r_old = zeros(n)                   # residual r_k
    s_old = zeros(n)
    XtR1 = zeros(m,m)
    
    ind = []
    # Difference buffers (capacity m columns)
    ΔF = zeros(T, n, m)        # ΔF[:, j] = F[:, j] − F[:, j+1]
    ΔR = similar(ΔF)           # ΔR[:, j] = R[:, j] − R[:, j+1]
    ΔX = similar(ΔF)           # ΔX[:, j] = X[:, j] − X[:, j+1]
    rhs = similar(s)

    sAA = similar(s)
    iters = 0
    for k in 1:maxiter
        iters = k
        
        j = (k+m-2) % m + 1

        # Convergence check
        if norm(s.- f, Inf) ≤ tol
            return f, iters, true
        end

        # Default next input is the plain fixed-point step (no acceleration).
        sAA .= f

        if k >= 2
            
            if k <= m+1
                ind = append!(ind, k-1)
            else
                ind[j] = k-1
            end
            
            ind1 = sortperm(ind; rev=true)
            ΔR[:,j] = r  - r_old
            ΔF[:,j] = f  - f_old
            ΔX[:,j] = s  - s_old    #  ΔX = ΔF - ΔR
                    
            @views ΔRsub = ΔR[:,ind1]
            @views ΔFsub = ΔF[:,ind1]
            @views ΔXsub = ΔX[:,ind1]
            
            mul!(XtR1, transpose(ΔX), ΔR)
            XtR = XtR1[ind1,ind1]
            @views XtR[diagind(XtR)] .+= λ
#             mul!(rhs, transpose(ΔXsub), r)
            rhs = ΔXsub' * r

            γ = try
                XtR \ rhs
            catch
                fill!(rhs, zero(eltype(rhs)))  # emergency fallback
                rhs
            end

            # Anderson update: s_{k+1}^AA = f_k − τ ΔF γ
            copyto!(sAA, f)
            mul!(sAA, ΔFsub, γ, -τ, one(T))            # sAA = 1*sAA + (-τ)*(Fsub*γ) 
        end

        # Advance to next iteration: recompute f_{k+1}, r_{k+1}
        f_old .= f
        r_old .= r
        s_old .= s
        s .= sAA   ## x_{i+1}
        f = G(s)     ## f(x_{i+1})
        r .= f .- s  ## is f(x_{i+1}) - x_{i+1}
    end
    
    return f, iters, false
end

function newton_lambda(C, y::AbstractVector;
        mode::Symbol = :halfspace,               # :halfspace or :ball_l2
        α::Union{Nothing,AbstractVector}=nothing,
        b::Union{Nothing,Real}=nothing,
        c::Union{Nothing,AbstractVector}=nothing,
        r::Union{Nothing,Real}=nothing,
        λ0::Real=0.0, tol::Real=1e-10, maxiter::Int=100, verbose::Bool=false)

    proj = make_projector(C)

    # ---- define φ(λ), y_of(λ), domain check, and residual at the final y_new ----
    local φ::Function, y_of::Function, dom_ok::Function, resid_at::Function

    if mode === :halfspace
        @assert α !== nothing && b !== nothing "mode=:halfspace requires α and b"
        local f1 = λ -> y .- λ .* α
        φ        = λ -> dot(α, proj(f1(λ))) - b
        y_of     = λ -> proj(f1(λ))
        dom_ok   = _  -> true
        resid_at = ŷ -> abs(dot(α, ŷ) - b)

    elseif mode === :ball_l2
        @assert c !== nothing && r !== nothing "mode=:ball_l2 requires c and r"
        local f1 = λ -> (y .+ λ .* c) ./ (1 + λ)     # requires 1+λ > 0
        φ        = λ -> 0.5 * norm(c .- proj(f1(λ)))^2 - 0.5 * r^2
        y_of     = λ -> proj(f1(λ))
        dom_ok   = λ -> (1 + λ) > 0
        resid_at = ŷ -> abs(0.5*norm(c .- ŷ)^2 - 0.5*r^2)
        
    else
        error("unknown mode = $mode; use :halfspace or :ball_l2")
    end

    # ---- Newton with backtracking (Armijo-like on |φ|) ----
    λ = Inf
    λnew = λ0
    @assert dom_ok(λ) "initial λ0 is outside domain (need 1+λ0>0 for :ball_l2)"
    g  = φ(λnew)
    it = 0

    for k in 0:maxiter
        it = k
        
        # convergence tests (residual and parameter change)
        if abs(g) ≤ tol || abs(λnew - λ) ≤ sqrt(eps(real(λ))) * (1 + abs(λ))
            y_new = y_of(λnew)
            return (y_new, λnew, resid_at(y_new), it, :converged)
        end
        
        λ = λnew
        
        if mode === :ball_l2
            gp0 = ForwardDiff.derivative(y_of, λ)
            x = y_of(λ)
            g = (norm(x)^2 - r^2) / 2
            gp = dot(x, gp0)
        else
            g  = φ(λ)
            gp = ForwardDiff.derivative(φ, λ)
        end
        
        if !isfinite(gp) || abs(gp) < 1e-14
            gp = sign(g) * max(abs(g), 1.0)
        end

        step  = g / gp
        λnew  = λ - step

        if verbose
            @info "iter=$k  λ=$λ  φ(λ)=$(g)  step=$(step)  →  λnew=$λnew  φ(λnew)=$(gnew)"
        end
        
    end

    # maxiter reached
    y_new = y_of(λ)
    return (y_new, λ, resid_at(y_new), it, :maxiter)
end

end