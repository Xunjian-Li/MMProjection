module BallFunctions

include("UtilityFunctions.jl") 
using LinearAlgebra
using ForwardDiff
using .UtilityFunctions

export project_with_balls_or_spheres, radii_from_target

"""
    lambda_step_once!(
        C, x, λw, W, c_list, r_list;
        mode=:ball, lambda_cap=1e6,
        armijo_c=1e-4, armijo_beta=0.5, t_min=1e-8
    )

Perform ONE projected Gauss–Newton step on active set `W`.

Inputs:
  - C           : constraint set for y (prox-available)
  - x           : original point
  - λw          : current λ on active set (length = |W|)
  - W           : active index set (Vector{Int})
  - c_list      : centers of balls
  - r_list      : radii of balls
  - mode        : `:ball` (λ ≥ 0) or `:sphere` (λ free)
  - lambda_cap  : upper bound on λ (applied to both modes if finite)

Output:
  (λw_new, y_new, F_new, accepted::Bool)

where
  F(λ) = 0.5 * ‖ψ(λ)‖²
  ψ_j(λ) = 0.5‖c_j - y(λ)‖² - 0.5 r_j²,  j ∈ W
"""



# ---------- y(λ) = Π_C( (x + Σ λ_j c_j) / (1 + Σ λ_j) ), scaled (supports λ<0) ----------
function y_of_balls(proj::Function,
                    x::AbstractVector,
                    c_list::AbstractVector{<:AbstractVector},
                    λvec::AbstractVector{<:Real})
    Tλ  = eltype(λvec)
    oneT = one(Tλ)

    # scale by max(1, max|λ|) to keep numbers O(1) even if λ huge or negative
    M = max(oneT, maximum(abs, λvec))
    invM = oneT / M

    s̃ = invM + sum(λvec) * invM             # s̃ = (1 + Σλ)/M
    # if s̃ too small, return a non-finite to trigger LM damping/step shrink
    if !isfinite(s̃) || abs(s̃) < Tλ(1e-14)
        return fill(Tλ(NaN), length(x))
    end

    num = x .* invM
    @inbounds for j in eachindex(c_list, λvec)
        num = num .+ ( (λvec[j]*invM) .* c_list[j] )
    end
    return proj(num ./ s̃)
end

function lambda_step_once!(
        C, x, λw::AbstractVector, W::Vector{Int},
        c_list::AbstractVector{<:AbstractVector},
        r_list::AbstractVector{<:Real};
        mode::Symbol=:ball,
        lambda_cap::Real=1e6,
        armijo_c::Real=1e-4,
        armijo_beta::Real=0.5,
        t_min::Real=1e-8
    )

    proj = make_projector(C)
    m = length(c_list)
    k = length(W)

    # --- helper: build full λ from λw on W ---
    function λ_full(λw_vec, Tλ=eltype(λw_vec))
        λ = zeros(Tλ, m)
        @inbounds for (i,j) in enumerate(W)
            λ[j] = λw_vec[i]
        end
        return λ
    end

    # --- helper: evaluate y(λ) ---
    function y_from(λw_vec)
        λ = λ_full(λw_vec)
        return y_of_balls(proj, x, c_list, λ)
    end

    # --- ψ(λw): residual vector on W ---
    function ψ(λw_vec)
        Tη = eltype(λw_vec)
        y  = y_from(λw_vec)
        out = similar(λw_vec)
        @inbounds for (i,j) in enumerate(W)
            out[i] = Tη(0.5)*sum(abs2, c_list[j] .- y) - Tη(0.5)*Tη(r_list[j]^2)
        end
        return out
    end

    # --- initial F, r, J, g ---
    r = ψ(λw)
    F = 0.5 * sum(abs2, r)
    J = ForwardDiff.jacobian(ψ, λw)
    g = J' * r               # gradient of F wrt λw

    # --- Gauss–Newton direction p = −(J'J)⁻¹ g ---
    JTJ = J' * J
    p = -(JTJ \ g)

    # --- project direction only for :ball case ---
    function project!(λv)
        if mode === :ball
            @inbounds for i in eachindex(λv)
                λv[i] = clamp(λv[i], 0.0, lambda_cap)
            end
        else
            if isfinite(lambda_cap)
                @inbounds for i in eachindex(λv)
                    λv[i] = clamp(λv[i], -lambda_cap, lambda_cap)
                end
            end
        end
        return λv
    end

    # --- Armijo line search ---
    gTp = dot(g, p)
    t   = 1.0
    accepted = false

    while t ≥ t_min
        λtrial = λw .+ t .* p
        project!(λtrial)

        # if nothing changed after projection, reduce t
        if norm(λtrial .- λw) < 1e-14
            t *= armijo_beta
            continue
        end

        rtrial = ψ(λtrial)
        Ftrial = 0.5 * sum(abs2, rtrial)

        if Ftrial ≤ F - armijo_c * t * abs(gTp)
            # success
            return (λtrial, y_from(λtrial), Ftrial, true)
        else
            t *= armijo_beta
        end
    end

    # --- fallback: no progress, return original ---
    return (λw, y_from(λw), F, false)
end


function project_with_balls_or_spheres(
    C, x, c_list, r_list;
    mode::Symbol=:ball, tol::Real=1e-8, max_outer::Int=30,
    lambda_cap::Real=1e3, δ_reg::Real=0.0, verbose::Bool=true)

    @assert mode in (:ball, :sphere)
    proj = make_projector(C)
    m = length(c_list)
    y = proj(x)
    λ = ones(m)

    residuals(y) = [0.5*sum(abs2, c_list[j] .- y) - 0.5*r_list[j]^2 for j in 1:m]
    ϕ = residuals(y)
    
    W = collect(1:m)
    

    iters = max_outer
    for outer in 1:max_outer
        iters = outer

        if verbose
            maxϕ = if isempty(W)
                -Inf
            elseif mode === :ball
                maximum(ϕ[W])
            else
                maximum(abs.(ϕ[W]))
            end
            @info "outer=$outer |W|=$(length(W)) maxφ=$(maxϕ)"
        end

        λw, y_new, F_new, ok = lambda_step_once!(
            C, x, λ[W], W, c_list, r_list;
            mode=mode, lambda_cap=lambda_cap
        )
        
        if norm(y_new.-y, Inf) < tol
            λ[W] .= λw
            return (y_new, λ, iters, :converged)
        end

        y .= y_new
        λ[W] .= λw

        ϕ = residuals(y)
        newW = if mode === :ball
            findall(j -> λ[j] > 1e-10, 1:m) |> collect
        else
            findall(j -> abs(λ[j]) > 1e-10, 1:m) |> collect
        end
        
        W = newW
    end

    return (y, λ, iters, :outer_maxiter)
end


# y* ∈ C
function radii_from_target(y_star::AbstractVector,
                           c_list::Vector{<:AbstractVector};
                           mode::Symbol = :ball,   # :ball or :sphere
                           slack::Real = 0.0)      # for ball constraint
    @assert mode in (:ball, :sphere)
    r_list = [norm(y_star .- c, 2) for c in c_list]
    if mode === :ball
        r_list .= r_list .+ max(slack, 0.0)
    end
    return r_list
end


end
