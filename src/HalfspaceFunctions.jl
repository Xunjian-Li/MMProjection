module HalfspaceFunctions

include("UtilityFunctions.jl") 
using LinearAlgebra
using ForwardDiff
using .UtilityFunctions

export project_halfspaces_active_set #, make_projector

########################  Projector from Prox  ########################

"""
    make_projector(C)

Creates a projection function `proj(u)` based on a proximal operator `prox(C, u)`.
Note: `prox(C, u)` must return `(proj_point, ...)`.

`proj(u) = argmin_y { I_C(y) + 0.5‖y-u‖² }`
"""
# make_projector(C) = (u -> prox(C, u)[1])


########################  Box-Specific ψ and Jacobian  ########################

"""
    ψ_and_jac_box!(y, ψw, Jw, x, A, b, W, λw, l, u)

Compute the residual `ψ(λw)` and its Jacobian `J` for the box constraint set `C = [l, u]`.

# Inputs
* `x`: Target point in R^n.
* `A`: Constraint matrix (size n×m).
* `b`: Constraint vector (length m).
* `W`: Active constraint indices (vector of Int).
* `λw`: Lagrange multipliers for the active set (length k = length(W)).
* `l`, `u`: Box bounds (scalar or AbstractVector).

# Outputs
* `y`: Projected point in R^n.
* `ψw`: Residual on the active set (length k).
* `Jw`: Jacobian matrix dψ/dλw (k×k).
"""
function ψ_and_jac_box!(
    y, ψw, Jw,
    x, A, b,
    W, λw,
    l, u,
)
    n, m = size(A)
    k = length(W)

    li(i) = l isa AbstractVector ? l[i] : l
    ui(i) = u isa AbstractVector ? u[i] : u

    #### 1) Construct full λ
    λ_full = zeros(eltype(λw), m)
    @inbounds for (i, j) in enumerate(W)
        λ_full[j] = λw[i]
    end

    #### 2) z = x - A * λ_full
    z = similar(x)
    mul!(z, A, λ_full)          # z = A * λ
    @inbounds for i in 1:n
        z[i] = x[i] - z[i]
    end

    #### 3) y = clamp.(z, l, u)
    @inbounds for i in 1:n
        zi  = z[i]
        lii = li(i)
        uii = ui(i)
        y[i] = zi < lii ? lii : (zi > uii ? uii : zi)
    end

    #### 4) mask: coordinates strictly inside the box (used for Diag(m))
    m_mask = falses(n)
    @inbounds for i in 1:n
        m_mask[i] = (z[i] > li(i)) & (z[i] < ui(i))
    end

    #### 5) ψ on active set: ψ_i = a_j' y - b_j
    @inbounds for (i, j) in enumerate(W)
        ψw[i] = dot(@view(A[:, j]), y) - b[j]
    end

    #### 6) Jacobian: Jw = -A_W' * Diag(m_mask) * A_W
    AW  = @view A[:, W]          # n×k
    tmp = similar(AW)            # n×k
    @inbounds for j in 1:k
        for i in 1:n
            tmp[i, j] = m_mask[i] ? AW[i, j] : zero(eltype(AW))
        end
    end

    mul!(Jw, transpose(AW), tmp) # Jw = AW' * tmp
    @. Jw = -Jw                  # -A_W' Diag(m) A_W

    return y, ψw, Jw
end

"""
    ψ_box_only!(y, ψw, x, A, b, W, λw, l, u)

Compute only the residual `ψ`, without the Jacobian. 
Used for repeated evaluation during line search.
"""
function ψ_box_only!(
    y, ψw,
    x, A, b,
    W, λw,
    l, u,
)
    n, m = size(A)
    k = length(W)

    li(i) = l isa AbstractVector ? l[i] : l
    ui(i) = u isa AbstractVector ? u[i] : u

    # Full λ
    λ_full = zeros(eltype(λw), m)
    @inbounds for (i, j) in enumerate(W)
        λ_full[j] = λw[i]
    end

    # z = x - A λ
    z = similar(x)
    mul!(z, A, λ_full)
    @inbounds for i in 1:n
        z[i] = x[i] - z[i]
    end

    # y = clamp.(z, l, u)
    @inbounds for i in 1:n
        zi  = z[i]
        lii = li(i)
        uii = ui(i)
        y[i] = zi < lii ? lii : (zi > uii ? uii : zi)
    end

    # ψ
    @inbounds for (i, j) in enumerate(W)
        ψw[i] = dot(@view(A[:, j]), y) - b[j]
    end

    return y, ψw
end


########################  Box-Specific Gauss–Newton  ########################

"""
    lambda_step_halfspace_box!(...)

Box-specialized Gauss–Newton step solver: 
`ψ_j(λ) = a_j' y(λ) - b_j`, where `y(λ) = Π_{[l,u]}(x - A λ)`.

# Arguments
- `mode = :halfspaces`: Enforces `λ ≥ 0`.
- `mode = :hyperplane`: `λ` is free.

# Returns
`(λw_new, y_new, F_new, accepted::Bool)`
"""
function lambda_step_halfspace_box!(
        x, λw::AbstractVector, W::Vector{Int},
        A::AbstractMatrix, b::AbstractVector;
        mode::Symbol=:halfspaces,
        armijo_c::Real=1e-8,
        armijo_beta::Real=0.5,
        t_min::Real=1e-8,
        tol::Real=1e-8,
        l=0.0, u=Inf,
    )

    @assert mode in (:halfspaces, :hyperplane)
    n, m = size(A)
    @assert length(x) == n
    @assert length(b) == m
    @assert all(1 .≤ W .≤ m)

    k  = length(λw)
    y  = zeros(eltype(x),  n)
    ψw = zeros(eltype(λw), k)
    Jw = zeros(eltype(λw), k, k)

    # Compute ψ and Jacobian J
    ψ_and_jac_box!(y, ψw, Jw, x, A, b, W, λw, l, u)

    r = ψw
    if norm(r, Inf) < tol
        Ftrial = 0.5 * sum(abs2, λw)
        return (λw, y, Ftrial, true)
    end
    F = 0.5 * sum(abs2, r)

    # Gauss–Newton: solve J p ≈ -r
    p = -(Jw \ r)

    # Helper: Projection for λ (non-negative if :halfspaces)
    project!(λv) = begin
        if mode === :halfspaces
            @inbounds for i in eachindex(λv)
                λv[i] = max(λv[i], zero(eltype(λv)))
            end
        end
        λv
    end

    g   = Jw' * r
    gTp = dot(g, p)

    # Armijo Line Search
    t = 1.0
    while t ≥ t_min

        λtrial = λw .+ t .* p
        project!(λtrial)

        if norm(λtrial .- λw) < 1e-14
            t *= armijo_beta
            continue
        end

        ytrial = similar(y)
        rtrial = similar(r)
        ψ_box_only!(ytrial, rtrial, x, A, b, W, λtrial, l, u)
        Ftrial = 0.5 * sum(abs2, rtrial)

        if Ftrial ≤ F - armijo_c * t * abs(gTp)
            return (λtrial, ytrial, Ftrial, true)
        end

        t *= armijo_beta
    end

    return (λw, y, 0.5 * sum(abs2, ψw), false)
end


########################  Generic Prox + AD Gauss–Newton  ########################

"""
    lambda_step_halfspace_prox!(...)

Generic Gauss–Newton step solver.
Obtains `y(λ) = proj(x - A λ)` via `proj = prox(C, ·)`, 
then performs Gauss–Newton on the active set residual `ψ_j(λ) = a_j' y(λ) - b_j`.

# Arguments
- `mode = :halfspaces`: Enforces `λ ≥ 0`.
- `mode = :hyperplane`: `λ` is free.
"""
function lambda_step_halfspace_prox!(
        C, x, λw::AbstractVector, W::Vector{Int},
        A::AbstractMatrix, b::AbstractVector;
        mode::Symbol=:halfspaces,
        armijo_c::Real=1e-8,
        armijo_beta::Real=0.5,
        t_min::Real=1e-8,
        tol::Real=1e-8,
    )

    @assert mode in (:halfspaces, :hyperplane)
    proj = make_projector(C)
    n, m = size(A)
    @assert length(x) == n
    @assert length(b) == m
    @assert all(1 .≤ W .≤ m)

    k  = length(λw)
    Jw = zeros(eltype(λw), k, k)

    # Helper to unpack λ to full size on active set
    function λ_full(λw_vec)
        λ = zeros(eltype(λw_vec), m)
        @inbounds for (i, j) in enumerate(W)
            λ[j] = λw_vec[i]
        end
        return λ
    end
    y_from(λw_vec) = proj(x - A * λ_full(λw_vec))

    # Residual ψ on W: ψ_j = a_j' y - b_j
    function ψ(λw_vec)
        y = y_from(λw_vec)
        out = similar(λw_vec)
        @inbounds for (i, j) in enumerate(W)
            out[i] = dot(A[:, j], y) - b[j]
        end
        return out
    end

    # Initial residual, objective, and Jacobian
    r = ψ(λw)
    if norm(r, Inf) < tol
        Ftrial = 0.5 * sum(abs2, λw)
        return (λw, y_from(λw), Ftrial, true)
    end
    F = 0.5 * sum(abs2, r)
    ForwardDiff.jacobian!(Jw, ψ, λw)

    # Gauss–Newton direction: solve J p ≈ -r (least squares via QR)
    p = -(qr(Jw) \ r)

    # Helper: Projection for λ
    function project!(λv)
        if mode === :halfspaces
            @inbounds for i in eachindex(λv)
                λv[i] = max(λv[i], 0.0)
            end
        end
        return λv
    end

    # Armijo backtracking on F
    g = Jw' * r
    gTp = dot(g, p)

    t = 1.0
    while t ≥ t_min
        λtrial = λw .+ t .* p
        project!(λtrial)
        if norm(λtrial .- λw) < 1e-14
            t *= armijo_beta
            continue
        end
        rtrial = ψ(λtrial)
        Ftrial = 0.5 * sum(abs2, rtrial)

        # Sufficient decrease check
        if Ftrial ≤ F - armijo_c * t * abs(gTp)
            return (λtrial, y_from(λtrial), Ftrial, true)
        end
        t *= armijo_beta
    end

    return (λw, y_from(λw), F, false)
end


########################  Unified Active-Set Projection Interface  ########################

"""
    project_halfspaces_active_set(
        C, x; A, b,
        mode      = :halfspaces,    # :halfspaces or :hyperplane
        use_box   = false,          # true -> use efficient box implementation
        l         = 0.0, u = Inf,   # box bounds (scalar or vector)
        λ_init    = nothing,
        tol       = 1e-8,
        max_outer = 30,
        verbose   = false,
    )

Performs active-set + Gauss–Newton projection onto the set:
    `{ y ∈ C : a_j' y ≤ b_j (or = b_j for :hyperplane) }`

- If `use_box = true`: Assumes `C = [l, u]` is a box. Uses `clamp` for projection and calculates explicit Jacobians (faster).
- If `use_box = false`: Obtains projector via `prox(C, ·)` and uses `ForwardDiff` for automatic Jacobian computation. Suitable for generic convex sets `C`.
"""
function project_halfspaces_active_set(
        C, x;
        A, b,
        mode::Symbol = :halfspaces,
        use_box::Bool = false,
        l = 0.0, u = Inf,
        λ_init = nothing,
        tol::Real = 1e-8,
        max_outer::Int = 30,
        verbose::Bool = false,
    )

    @assert mode in (:halfspaces, :hyperplane)
    n, m = size(A)
    @assert length(x) == n
    @assert length(b) == m

    # Define projector
    proj = if use_box
        # Use clamp directly for box projection
        function (u_vec)
            y = similar(u_vec)
            li(i) = l isa AbstractVector ? l[i] : l
            ui(i) = u isa AbstractVector ? u[i] : u
            @inbounds for i in eachindex(u_vec)
                zi  = u_vec[i]
                lii = li(i)
                uii = ui(i)
                y[i] = zi < lii ? lii : (zi > uii ? uii : zi)
            end
            y
        end
    else
        make_projector(C)
    end

    # Initial y, λ
    y = proj(x)
    λ = λ_init === nothing ? zeros(eltype(x), m) : copy(λ_init)

    ψ_full(y_) = [dot(A[:, j], y_) - b[j] for j in 1:m]
    ψ = ψ_full(y)

    # Initial active set
    if mode === :halfspaces
        W = findall(j -> ψ[j] > 1e-10, 1:m)
    else
        W = findall(j -> abs(ψ[j]) > 1e-10, 1:m)
    end

    iters = max_outer

    for outer in 1:max_outer
        iters = outer

        if verbose
            maxψ = isempty(W) ? -Inf :
                   (mode === :halfspaces ? maximum(ψ[W]) : maximum(abs.(ψ[W])))
            @info "outer=$outer  |W|=$(length(W))  maxψ=$(maxψ)"
        end

        if isempty(W)
            # Already feasible
            return (y, λ, iters, :converged)
        end

        λw_old = λ[W]

        if use_box
            λw_new, y_new, F_new, ok = lambda_step_halfspace_box!(
                x, λw_old, W, A, b;
                mode = mode, tol = tol, l = l, u = u,
            )
        else
            λw_new, y_new, F_new, ok = lambda_step_halfspace_prox!(
                C, x, λw_old, W, A, b;
                mode = mode, tol = tol,
            )
        end

        # Simple convergence criteria
        if sum((y_new .- y).^2) < tol || norm(ψ[W], 2) < tol
            λ[W] .= λw_new
            y .= y_new
            return (y, λ, iters, :converged)
        end

        λ[W] .= λw_new
        y .= y_new
        ψ .= ψ_full(y)

        # Update active set
        if mode === :halfspaces
            W = findall(j -> λ[j] > 1e-10, 1:m)
        else
            W = findall(j -> abs(λ[j]) > 1e-10, 1:m)
        end
    end

    return (y, λ, iters, :outer_maxiter)
end

end # module ActiveSetProjections