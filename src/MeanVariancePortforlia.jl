module MeanVariancePortforlia

include("HalfspaceFunctions.jl") 
using LinearAlgebra
using .HalfspaceFunctions

include("DykstraFuntions.jl")
using .DykstraFuntions

include("UtilityFunctions.jl")
using .UtilityFunctions


export mean_variance_mle_gradient


mutable struct MeanVarianceProblem{T}
    μ::Vector{T}
    Σ::Matrix{T}
    r::T
    max_iter::Int
    tol::T
    use_nesterov::Bool
    verbose::Bool

    x::Vector{T}
    x_prev::Vector{T}
    x_new::Vector{T}
    grad::Vector{T}
    y::Vector{T}
    buffer::Vector{T}
    scratch::Vector{T}
    scratch2::Vector{Int}
    p1::Vector{T}
    p2::Vector{T}

    λ::T
    Λ::Vector{T}
    loss_all::Vector{T}
end

function MeanVarianceProblem(μ::Vector{T}, Σ::Matrix{T}, r::T;
    max_iter::Int = 5000, tol::T = 1e-6, use_nesterov::Bool = false, verbose::Bool = false) where T

    n = length(μ)
    x = fill(one(T)/n, n)
    x_prev = copy(x)
    x_new = similar(x)
    grad = similar(x)
    y = copy(x)
    buffer = similar(x)
    scratch = similar(x)
    scratch2 = Vector{Int}(undef, size(x)...)
    p1 = zeros(T, n)
    p2 = zeros(T, n)

    λ = eigenvalue_max(Σ .- Diagonal(diag(Σ)))
    Λ = diag(Σ) .+ λ

    MeanVarianceProblem{T}(μ, Σ, r, max_iter, tol, use_nesterov, verbose,
        x, x_prev, x_new,
        grad, y, buffer, scratch, scratch2, p1, p2,
        λ, Λ, T[])
end

function eigenvalue_max(A::Matrix{T}; iters::Int = 10) where T
    n = size(A, 1)
    b = randn(T, n)
    for _ in 1:iters
        b = A * b
        b ./= norm(b)
    end
    return dot(b, A * b)
end

function mean_variance_mle_gradient(
        μ::Vector{T}, Σ::Matrix{T}, r::T;
        max_iter::Int = 5000, tol::T = 1e-6,
        use_nesterov::Bool = true, 
        verbose::Bool = false, 
        method::Symbol = :newton, #  AAI, dykstra, newton
        standard::Bool = true
    ) where T
    
    MV = MeanVarianceProblem(μ, Σ, r; max_iter=max_iter, tol=tol, use_nesterov=use_nesterov, verbose=verbose)
    
    t = one(T)
    n = length(μ)
    d = diag(Σ)
    
    D_sqrt = sqrt.(d)
    D_inv_sqrt = 1.0 ./ D_sqrt
    Σ_0 = Diagonal(D_inv_sqrt) * Σ * Diagonal(D_inv_sqrt)
    λ = eigenvalue_max(Σ_0)
    μ_tilde = D_inv_sqrt .* μ
    λ₀ = zeros(1)
    Mat_μ = Matrix(reshape(-μ_tilde, length(μ_tilde), 1))
    
    mul!(MV.grad, Σ_0, MV.y)
    loss = dot(MV.grad, MV.y)
    push!(MV.loss_all, loss)
    
    iters = MV.max_iter
    
    for iter in 1:MV.max_iter
        
        @. MV.buffer = MV.y - MV.grad / λ
        w = collect(D_inv_sqrt)
        C = IndWSimplex(w, 1.0)
        
        if method === :newton
            reng = -r
            kwargs2 = (
                A=Mat_μ, 
                b = [reng], 
                mode=:halfspaces,
                tol=1e-12, 
                λ_init = λ₀,
                max_outer=100
            )
            MV.x_new, λ₀, _, _ = project_halfspaces_active_set(C, MV.buffer; kwargs2...)
        else
            kwargs = (
                m        = 4,      # memory
                tol      = 1e-8,   # stopping tolerance
                τ        = 1.0,    # damping
                maxiter  = 2000,
                λ        = 1e-12,
                cnd_max  = 1e8,
            )
            specs = [
                (type=:halfspace, α=-μ_tilde, c=-r),
                (type=:wsimplex, w=w,  b=1.0),
            ]
            rLambda = zeros(n, length(specs));
            x0 = similar(μ_tilde);
            sets = build_sets(specs; p=length(specs))
            
            if method === :dykstra
                MV.x_new, Λ_Dy, iters_Dy, ok_Dy = run_dykstra!(x0, 
                                MV.buffer, rLambda, sets; AAvariant  = :none, kwargs...)
            elseif method === :AAI
                MV.x_new, Λ_AAI, iters_AAI, ok_AAI = run_dykstra!(x0, 
                                MV.buffer, rLambda, sets; AAvariant  = :AAI, kwargs...)
            end
        end

        if norm(MV.x_new - MV.x_prev) < MV.tol * length(MV.x_prev)
            MV.verbose && println("Converged at iteration $iter")
            iters = iter
            break
        end
        
        if MV.use_nesterov
            t_next = (one(T) + sqrt(one(T) + 4 * t^2)) / 2
            β = (t - one(T)) / t_next
            @. MV.scratch = MV.x_new + β * (MV.x_new - MV.x_prev)

            mul!(MV.grad, Σ_0, MV.scratch)
            loss_nest = dot(MV.grad, MV.scratch)
            
            if loss_nest < loss
                copyto!(MV.y, MV.scratch)
                t = t_next
                loss = loss_nest
            else
                copyto!(MV.y, MV.x_new)
                mul!(MV.grad, Σ_0, MV.y)
                t = one(T)
            end
        else
            copyto!(MV.y, MV.x_new)
            mul!(MV.grad, Σ_0, MV.y)
            loss = dot(MV.grad, MV.x_new)
        end
        push!(MV.loss_all, loss)
        copyto!(MV.x_prev, MV.x_new)
    end

    weights = MV.x_prev .* D_inv_sqrt
    port_return = dot(MV.μ, weights)
    port_variance = dot(weights, MV.Σ * weights)
    port_risk = sqrt(port_variance)

    return weights, port_variance, port_return, port_risk, iters, MV.loss_all
end


end