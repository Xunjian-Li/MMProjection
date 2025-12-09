module OptimalTransportSolvers

using LinearAlgebra
using Statistics
using Printf
using Base.Threads
using LoopVectorization

export solve_mm_ot

# greet() = print("Hello from Dykstra Package!")

# ==============================================================================
# 1. Unified Workspace
# ==============================================================================

struct OTWorkspace{T <: AbstractFloat}
    m::Int; p::Int; n_vars::Int
    
    # -- Matrices (9) --
    T_mat::Matrix{T}      # 4
    T_prev::Matrix{T}     # 5
    Y_mat::Matrix{T}      # 6
    Z::Matrix{T}          # 7
    Z_t::Matrix{T}        # 8
    J::Matrix{T}          # 9
    H::Matrix{T}          # 10
    S::Matrix{T}          # 11
    Y::Matrix{T}          # 12
    
    # -- Vectors (10) --
    rho::Vector{T}        # 13
    alpha::Vector{T}      # 14
    q::Vector{T}          # 15
    v_new::Vector{T}      # 16
    F_new::Vector{T}      # 17
    v::Vector{T}          # 18
    v_old::Vector{T}      # 19
    F::Vector{T}          # 20
    g::Vector{T}          # 21
    d::Vector{T}          # 22
    
    function OTWorkspace(m::Int, p::Int; T_type=Float64, history_size=10)
        n = m + p
        new{T_type}(
            m, p, n,                                    # 1, 2, 3
            
            # Matrices (9)
            Matrix{T_type}(undef, m, p),                # 4. T_mat
            Matrix{T_type}(undef, m, p),                # 5. T_prev
            Matrix{T_type}(undef, m, p),                # 6. Y_mat
            Matrix{T_type}(undef, m, p),                # 7. Z
            Matrix{T_type}(undef, p, m),                # 8. Z_t
            zeros(T_type, n, n),                        # 9. J
            zeros(T_type, n, n),                        # 10. H
            Matrix{T_type}(undef, n, history_size),     # 11. S
            Matrix{T_type}(undef, n, history_size),     # 12. Y
            
            # Vectors (10)
            Vector{T_type}(undef, history_size),        # 13. rho
            Vector{T_type}(undef, history_size),        # 14. alpha
            Vector{T_type}(undef, n),                   # 15. q
            Vector{T_type}(undef, n),                   # 16. v_new
            Vector{T_type}(undef, n),                   # 17. F_new
            zeros(T_type, n),                           # 18. v
            zeros(T_type, n),                           # 19. v_old
            Vector{T_type}(undef, n),                   # 20. F
            zeros(T_type, n),                           # 21. g
            zeros(T_type, n)                            # 22. d
        )
    end
end

# ==============================================================================
# 2. Inner Solver: Coordinate Descent (Standard & Exact)
# ==============================================================================

function solve_bcd_kernel!(work::OTWorkspace, a, b, tol, max_iter, mode::Symbol)
    m, p = work.m, work.p
    lambda = view(work.v, 1:m)
    theta  = view(work.v, m+1:m+p)
    n_threads = Threads.nthreads()
    max_ups = zeros(Float64, n_threads)
    row_active = fill(true, m); col_active = fill(true, p)
    scan_int = 10
    is_exact = (mode == :exact)
    sub_max = is_exact ? 20 : 1
    sub_tol = 1e-10

    iter = 0; converged = false

    for k in 1:max_iter
        iter += 1
        is_full = (k==1 || k%scan_int==0)
        fill!(max_ups, 0.0)

        # Row Update
        Threads.@threads for i in 1:m
            if !is_full && !row_active[i]; continue; end
            tid = Threads.threadid(); lam_i = lambda[i]; initial_val = lam_i
            Zc = view(work.Z_t, :, i)
            
            for _ in 1:sub_max
                r_sum = 0.0; cnt = 0; max_v = -Inf
                @turbo for j in 1:p
                    pot = Zc[j] - theta[j]
                    max_v = max(max_v, pot)
                    act = pot > lam_i
                    r_sum += (pot - lam_i) * act
                    cnt += act
                end
                res = r_sum - a[i]
                if is_exact && abs(res) < sub_tol; break; end
                if cnt > 0
                    step = res/cnt; lam_i += step
                    if is_exact && abs(step) < 1e-13; break; end
                else
                    lam_i = max_v - 1e-9; break
                end
            end
            lambda[i] = lam_i
            loc_up = abs(lam_i - initial_val)
            row_active[i] = (loc_up > tol)
            if loc_up > max_ups[tid]; max_ups[tid] = loc_up; end
        end

        # Col Update
        Threads.@threads for j in 1:p
            if !is_full && !col_active[j]; continue; end
            tid = Threads.threadid(); th_j = theta[j]; initial_val = th_j
            Zc = view(work.Z, :, j)
            
            for _ in 1:sub_max
                c_sum = 0.0; cnt = 0; max_v = -Inf
                @turbo for i in 1:m
                    pot = Zc[i] - lambda[i]
                    max_v = max(max_v, pot)
                    act = pot > th_j
                    c_sum += (pot - th_j) * act
                    cnt += act
                end
                res = c_sum - b[j]
                if is_exact && abs(res) < sub_tol; break; end
                if cnt > 0
                    step = res/cnt; th_j += step
                    if is_exact && abs(step) < 1e-13; break; end
                else
                    th_j = max_v - 1e-9; break
                end
            end
            theta[j] = th_j
            loc_up = abs(th_j - initial_val)
            col_active[j] = (loc_up > tol)
            if loc_up > max_ups[tid]; max_ups[tid] = loc_up; end
        end

        if is_full && maximum(max_ups) < tol; converged = true; break; end
    end
    return converged, iter, 0
end

# ==============================================================================
# 3. Inner Solver: Robust Global Newton
# ==============================================================================

function compute_newton_sys!(w::OTWorkspace, a, b; calc_J=true)
    fill!(w.F, 0.0); if calc_J; fill!(w.J, 0.0); end
    m, p = w.m, w.p
    @inbounds for j in 1:p
        th = w.v[m+j]
        for i in 1:m
            val = w.Z[i,j] - w.v[i] - th
            if val > 0
                w.F[i] += val; w.F[m+j] += val
                if calc_J
                    w.J[i,i] -= 1; w.J[m+j,m+j] -= 1
                    w.J[i,m+j] = -1; w.J[m+j,i] = -1
                end
            end
        end
    end
    @. w.F[1:m] -= a; @. w.F[m+1:end] -= b
    return 0.5 * dot(w.F, w.F)
end

function solve_newton_kernel!(work::OTWorkspace, a, b, tol, max_iter)
    n = work.n_vars
    Phi = compute_newton_sys!(work, a, b; calc_J=true)
    rescue_count = 0

    for it in 1:max_iter
        if norm(work.F, Inf) < tol; return true, it, rescue_count; end

        mul!(work.g, work.J, work.F)
        is_stuck = (norm(work.g) < 1e-9) && (norm(work.F, Inf) > 10 * tol)
        
        if is_stuck
            copyto!(work.d, work.F); rescue_count += 1
        else
            mul!(work.H, work.J', work.J)
            @inbounds for i in 1:n; work.H[i, i] += 1e-4; end
            work.d .= -(work.H \ work.g)
            if dot(work.g, work.d) >= 0; @. work.d = -work.g; end
        end

        t = 1.0; copyto!(work.v_old, work.v); Phi_old = Phi
        accepted = false
        
        for ls in 1:10
            @. work.v = work.v_old + t * work.d
            Phi_new = compute_newton_sys!(work, a, b; calc_J=false)
            if Phi_new < Phi_old; Phi = Phi_new; accepted = true; break; end
            t *= 0.5
        end

        if accepted
            Phi = compute_newton_sys!(work, a, b; calc_J=true)
        else
            @. work.v = work.v_old + 1e-4 * work.F
            Phi = compute_newton_sys!(work, a, b; calc_J=true)
        end
    end
    return false, max_iter, rescue_count
end

# ==============================================================================
# 4. Inner Solver: L-BFGS
# ==============================================================================

function compute_lbfgs_sys!(work::OTWorkspace, v_curr::AbstractVector, F_out::AbstractVector, a, b)
    m, p = work.m, work.p
    fill!(F_out, 0.0); squared_norm_T = 0.0
    
    @inbounds for j in 1:p
        theta_val = v_curr[m+j]
        for i in 1:m
            val = work.Z[i, j] - v_curr[i] - theta_val
            if val > 0
                work.T_mat[i, j] = val
                F_out[i] += val; F_out[m + j] += val
                squared_norm_T += val^2
            else
                work.T_mat[i, j] = 0.0
            end
        end
    end
    @inbounds for i in 1:m; F_out[i] -= a[i]; end
    @inbounds for j in 1:p; F_out[m + j] -= b[j]; end
    
    dot_lam = dot(view(v_curr, 1:m), a)
    dot_th = dot(view(v_curr, m+1:m+p), b)
    return 0.5 * squared_norm_T + dot_lam + dot_th
end

function solve_lbfgs_kernel!(work::OTWorkspace, a, b, tol_grad, max_iter)
    n = work.n_vars
    hist_size = length(work.rho)
    
    # Initial Eval
    f_val = compute_lbfgs_sys!(work, work.v, work.F, a, b)
    head = 1; len = 0
    
    for k in 1:max_iter
        if norm(work.F, Inf) < tol_grad; return true, k, 0; end
        
        copyto!(work.q, work.F)
        
        curr = head - 1; if curr < 1; curr = hist_size; end
        for _ in 1:len
            sq = dot(view(work.S, :, curr), work.q)
            al = work.rho[curr] * sq
            work.alpha[curr] = al
            @inbounds for i in 1:n; work.q[i] -= al * work.Y[i, curr]; end
            curr -= 1; if curr < 1; curr = hist_size; end
        end
        
        if len > 0
            last = head - 1; if last < 1; last = hist_size; end
            sy = dot(view(work.S, :, last), view(work.Y, :, last))
            yy = dot(view(work.Y, :, last), view(work.Y, :, last))
            if yy > 1e-10; work.q .*= (sy / yy); end
        end
        
        start_idx = head - len; if start_idx < 1; start_idx += hist_size; end
        curr = start_idx
        for _ in 1:len
            yq = dot(view(work.Y, :, curr), work.q)
            beta = work.rho[curr] * yq
            @inbounds for i in 1:n; work.q[i] += (work.alpha[curr] - beta) * work.S[i, curr]; end
            curr += 1; if curr > hist_size; curr = 1; end
        end
        
        # [Corrected Direction] d = q (Since F is negative gradient)
        copyto!(work.d, work.q)
        
        gtd = -dot(work.F, work.d)
        if gtd >= 0; @. work.d = work.F; gtd = -dot(work.F, work.F); len = 0; end
        
        step = 1.0; accepted = false; c1 = 1e-4
        for _ in 1:10
            @. work.v_new = work.v + step * work.d
            f_new = compute_lbfgs_sys!(work, work.v_new, work.F_new, a, b)
            if f_new <= f_val + c1 * step * gtd; f_val = f_new; accepted = true; break; end
            step *= 0.5
        end
        
        if !accepted; step=1e-3; @. work.v_new = work.v + step*work.d; f_val = compute_lbfgs_sys!(work, work.v_new, work.F_new, a, b); end
        
        @. work.S[:, head] = step * work.d 
        @. work.Y[:, head] = work.F - work.F_new
        
        ys = dot(view(work.Y, :, head), view(work.S, :, head))
        if ys > 1e-10
            work.rho[head] = 1.0 / ys
            head = (head % hist_size) + 1
            len = min(len + 1, hist_size)
        end
        
        copyto!(work.v, work.v_new)
        copyto!(work.F, work.F_new)
    end
    return false, max_iter, 0
end

# ==============================================================================
# 5. Main MM Solver (Unified)
# ==============================================================================

function solve_mm_ot(
    C::Matrix{T}, a::Vector{T}, b::Vector{T}; 
    e::Float64 = 100.0, 
    method::Symbol = :standard, 
    use_fista::Bool = true, 
    max_outer::Int = 1000, 
    tol_outer::Float64 = 1e-8,
    max_inner::Int = 100, 
    tol_inner::Float64 = 1e-10,
    verbose::Bool = true
) where T <: AbstractFloat

    m, p = size(C)
    work = OTWorkspace(m, p; T_type=T)
    mul!(work.T_mat, a, b')
    
    if use_fista
        copyto!(work.T_prev, work.T_mat); copyto!(work.Y_mat, work.T_mat)
    end
    fill!(work.v, 0.0)
    
    obj_hist = Float64[]
    inv_e = 1.0 / e
    t_k = 1.0
    final_iters = max_outer
    thread_costs = zeros(Threads.nthreads())

    if verbose
        m_str = uppercase(string(method))
        f_str = use_fista ? "ON" : "OFF"
        println("\n" * "="^80)
        println("  MM-OT Solver (Method: $m_str | FISTA: $f_str | Threads: $(Threads.nthreads()))")
        println("="^80)
        @printf("%-6s | %-14s | %-12s | %-8s | %-s\n", "Iter", "Cost", "Violation", "InnerIt", "Note")
        println("-"^80)
    end

    for k in 1:max_outer
        if use_fista
            if k > 1
                t_n = (1+sqrt(1+4*t_k^2))/2; beta = (t_k-1)/t_n
                @turbo for i in eachindex(work.Y_mat)
                    work.Y_mat[i] = work.T_mat[i] + beta * (work.T_mat[i] - work.T_prev[i])
                end
                copyto!(work.T_prev, work.T_mat); t_k = t_n
            else; copyto!(work.Y_mat, work.T_mat); end
            @turbo for i in eachindex(work.Z); work.Z[i] = work.Y_mat[i] - C[i]*inv_e; end
        else
            @turbo for i in eachindex(work.Z); work.Z[i] = work.T_mat[i] - C[i]*inv_e; end
        end

        inner_it = 0; rescues = 0
        
        if method == :newton
            _, inner_it, rescues = solve_newton_kernel!(work, a, b, tol_inner, 100)
        elseif method == :lbfgs
            _, inner_it, _ = solve_lbfgs_kernel!(work, a, b, tol_inner, 100)
        else 
            transpose!(work.Z_t, work.Z)
            _, inner_it, _ = solve_bcd_kernel!(work, a, b, tol_inner, max_inner, method)
        end

        fill!(thread_costs, 0.0)
        lambda = view(work.v, 1:m); theta = view(work.v, m+1:m+p)
        Threads.@threads for j in 1:p
            tid = Threads.threadid(); loc_c = 0.0
            Zc = view(work.Z, :, j); Tc = view(work.T_mat, :, j); Cc = view(C, :, j)
            th = theta[j]
            @turbo for i in 1:m
                v = Zc[i] - lambda[i] - th
                vc = v * (v > 0)
                Tc[i] = vc
                loc_c += Cc[i] * vc
            end
            thread_costs[tid] += loc_c
        end
        lin_cost = sum(thread_costs)
        
        sum!(view(work.F, 1:m), work.T_mat) 
        col_sums = sum(work.T_mat, dims=1)
        copyto!(view(work.F, m+1:m+p), vec(col_sums))
        @. work.F[1:m] -= a; @. work.F[m+1:end] -= b
        viol = norm(work.F, Inf)
        
        push!(obj_hist, lin_cost)
        note = rescues > 0 ? "Rescues: $rescues" : ""

        if verbose && (k <= 10 || k % 50 == 0)
            @printf("%-6d | %-14.8e | %-12.2e | %-8d | %s\n", k, lin_cost, viol, inner_it, note)
        end
        
        if k > 1 
            obj_diff = abs(obj_hist[end]-obj_hist[end-1])
            if obj_diff < tol_outer && viol < 1e-6
                 if verbose
                    println("-"^80)
                    println(">> Converged at iteration $k")
                 end
                 final_iters = k
                 break
            end
        end
    end
    
    return work.T_mat, final_iters, obj_hist, inv_e
end

end # module