module WeightedSimplex

using LoopVectorization
using Random
using Base.Threads
using BangBang
using SparseArrays

include("UtilityFunctions.jl")
using .UtilityFunctions


export Simplex_MM_Projection, wcondat_p, reservoir_indices!

##############################
# Weighted Simplex Projection
#   min 0.5 * ||x - y||^2
#   s.t. x >= 0,  sum(w .* x) = r
#
# Parallel MM algorithm with active-set shrinking.
##############################

function filter_inplace!(idx::Vector{Int}, n::Int)
    write = 0
    @inbounds for read in 1:length(idx)
        i = idx[read]
        if i <= n
            write += 1
            idx[write] = i
        end
    end
    resize!(idx, write)
    return idx
end

function stream_lambda_on_sample(y, w, b, sample)
    @assert !isempty(sample)
    n = length(y)
    idx = filter_inplace!(sample, n)
    b *= length(idx)/n
    
    s1 = y[idx[1]]*w[idx[1]]
    s2 = w[idx[1]]^2
    λ  = (s1 - b)/s2
    active = Int[idx[1]]
    wait   = Int[]
    @inbounds for t in 2:length(idx)
        i = idx[t]; yi = y[i]; wi = w[i]
        if yi/wi > λ
            λ_try = (s1 + yi*wi - b)/(s2 + wi^2)
            if λ_try > (yi*wi - b)/(wi^2)
                push!(active, i); s1 += yi*wi; s2 += wi^2; λ = (s1 - b)/s2
            else
                append!(wait, active); empty!(active)
                push!(active, i); s1 = yi*wi; s2 = wi^2; λ = (s1 - b)/s2
            end
        end
    end
    
    @inbounds for j in wait
        if y[j]/w[j] > λ
            push!(active, j); s1 += y[j]*w[j]; s2 += w[j]^2; λ = (s1 - b)/s2
        end
    end
    return λ
end

function reservoir_indices!(out::Vector{Int}, m::Int, n::Int; rng=Random.default_rng())
    empty!(out)
    m ≤ 0 && return out
    resize!(out, min(m, n))
    @inbounds for i in 1:min(m, n)
        out[i] = i
    end
    i = m + 1
    @inbounds while i ≤ n
        j = rand(rng, 1:i)
        if j ≤ m
            out[j] = i
        end
        i += 1
    end
    return out
end

# ================================================================
# Parallel, stable (order-preserving) in-place compaction.
#
# Given an active set A and threshold λ, keep indices satisfying
#   y[i]/w[i] > λ   (numerically: y[i] >= λ*w[i] - tol).
#
# At the same time, accumulate
#   s1 = ∑ w[i]*y[i],   s2 = ∑ w[i]^2
# over the retained indices.
#
# Returns:
#   newlen   = new length of A
#   s1, s2   = updated sums
#
# Note: block-wise parallel filtering, but write-back is sequential
#       to preserve global ordering.
# ================================================================
function parallel_filter_compact!(A::Vector{Int},
                                   y::AbstractVector{T},
                                   w::AbstractVector{T},
                                   λ::T;
                                   tol::T = sqrt(eps(T))) where {T<:Real}
    nA = length(A)
    nt = Threads.nthreads()
    locals = Vector{Vector{Int}}(undef, nt)
    s1loc  = zeros(T, nt)
    s2loc  = zeros(T, nt)

    # Pass 1: thread-local filtering (each thread processes a contiguous block).
    @threads for t in 1:nt
        i0 = fld((t-1)*nA, nt) + 1
        i1 = fld(t*nA, nt)
        buf = Vector{Int}()
        @inbounds for k in i0:i1
            i  = A[k]
            yi = y[i]; wi = w[i]
            thresh = λ * wi
            keep = yi >= (thresh - tol * (abs(yi) + abs(thresh) + one(T)))
            if keep
                push!(buf, i)
                s1loc[t] += wi * yi
                s2loc[t] += wi * wi
            end
        end
        locals[t] = buf
    end

    # Compute prefix offsets for each thread block
    sizes = map(length, locals)
    offs  = Vector{Int}(undef, nt)
    total = 0
    @inbounds for t in 1:nt
        offs[t] = total + 1
        total  += sizes[t]
    end

    # Pass 2: sequential write-back to A (preserves block order).
    @inbounds for t in 1:nt
        buf = locals[t]
        off = offs[t]
        @simd for j in 1:length(buf)
            A[off + j - 1] = buf[j]
        end
    end

    # Reduce thread-local sums
    s1 = zero(T); s2 = zero(T)
    @inbounds for t in 1:nt
        s1 += s1loc[t];  s2 += s2loc[t]
    end

    return total, s1, s2
end

# ================================================================
# Sequential coordinate-style shrinking.
#
# One-pass scan: elements violating y[i] <= λ*w[i] are removed
# immediately, and (s1,s2,λ) are updated in-place.
#
# Returns:
#   newlen   = new active set size
#   s1, s2   = updated sums
#   λ        = updated pivot
#
# This mimics the behavior of wcheckL but in-place and zero-allocation.
# ================================================================
function shrink_sequential_coord!(A::Vector{Int},
                                  y::AbstractVector{T},
                                  w::AbstractVector{T},
                                  r::T,
                                  λ::T,
                                  s1::T,
                                  s2::T;
                                  tol::T = sqrt(eps(T))) where {T<:Real}
    write = 0
    nA = length(A)
    @inbounds for read in 1:nA
        i  = A[read]
        yi = y[i]; wi = w[i]
        thresh = λ * wi
        keep = yi > (thresh + tol * (abs(yi) + abs(thresh) + one(T)))
        if keep
            write += 1
            A[write] = i
        else
            s1 -= wi * yi
            s2 -= wi * wi
            if s2 <= zero(T)
                resize!(A, 0)
                return 0, s1, s2, λ
            end
            λ = (s1 - r) / s2
        end
    end
    resize!(A, write)
    return write, s1, s2, λ
end

# ================================================================
# Adaptive shrinking step.
#
# If active set is large: run parallel compaction (freezes λ).
# If active set is small: run coordinate-style sequential shrinking.
#
# Returns: newlen, s1, s2, λ
# ================================================================
function shrink_step_adaptive!(A::Vector{Int},
                               y::AbstractVector{T},
                               w::AbstractVector{T},
                               r::T,
                               λ::T,
                               s1::T,
                               s2::T;
                               thresh::Int = 200_000) where {T<:Real}
    if length(A) > thresh
        newlen, s1p, s2p = parallel_filter_compact!(A, y, w, λ)
        resize!(A, newlen)
        λp = (s1p - r) / s2p
        return newlen, s1p, s2p, λp
    else
        newlen, s1p, s2p, λp = shrink_sequential_coord!(A, y, w, r, λ, s1, s2)
        resize!(A, newlen)
        return newlen, s1p, s2p, λp
    end
end


function active_set_initial(y::AbstractVector{T}, w::AbstractVector{T}, λ::T) where {T<:Real}
    n  = length(y)
    nt = nthreads()
    locals = Vector{Vector{Int}}(undef, nt)
    s1loc  = zeros(T, nt)
    s2loc  = zeros(T, nt)

    @threads for t in 1:nt
        i0 = fld((t-1)*n, nt) + 1
        i1 = fld(t*n, nt)
        buf = Vector{Int}()
        sizehint!(buf, max(8, (i1-i0+1)÷5))
        s1t = zero(T); s2t = zero(T)
        @inbounds for i in i0:i1
            yi = y[i]; wi = w[i]
            if yi > λ * wi
                push!(buf, i)
                s1t += wi * yi
                s2t += wi * wi
            end
        end
        locals[t] = buf
        s1loc[t]  = s1t
        s2loc[t]  = s2t
    end
    offs  = Vector{Int}(undef, nt)
    total = 0
    @inbounds for t in 1:nt
        offs[t] = total + 1
        total  += length(locals[t])
    end
    A = Vector{Int}(undef, total)
    @inbounds for t in 1:nt
        buf = locals[t]; off = offs[t]
        @simd for j in 1:length(buf)
            A[off + j - 1] = buf[j]
        end
    end

    s1 = zero(T); s2 = zero(T)
    @inbounds for t in 1:nt
        s1 += s1loc[t]; s2 += s2loc[t]
    end
    return A, s1, s2
end

# ================================================================
# Main solver: MM on λ with active-set shrinking.
#
# If use_wfilter==false, start from A = 1:n (fully active).
# Supports extrapolation with rollback if overshoot occurs.
#
# Returns:
#   x      = sparse solution vector
#   λ      = final dual variable
#   iters  = number of iterations
# ================================================================
function Simplex_MM_Projection(y::AbstractVector{T},
                                           w::AbstractVector{T},
                                           r::T;
                                           tol::T = T(1e-10),
                                           maxit::Int = 1_000,  
                                           sample::AbstractVector{Int} = DEFAULT_SAMPLE,
                                           extrap::Bool = false,
                                           extrap_factor::T = T(4)) where {T<:Real}
    n  = length(y)
    
    λ = stream_lambda_on_sample(y, w, r, sample)
    A, s1, s2 = active_set_initial(y, w, λ)
    
    λ_new = (s1-r)/s2
    λ = λ_new
    
    iters = 0
    
    if extrap
        A_safe = similar(A); 
        resize!(A_safe, length(A))
        copyto!(A_safe, 1, A, 1, length(A))
        lenA = length(A); lenA_safe = length(A_safe)
        s1_safe = s1; s2_safe = s2; λ_safe = λ
        extraploop = true
    else
        extraploop = false
    end
    
    sum_A = 0
    for it in 1:maxit
        iters = it
        sum_A += length(A)
        
        newlen, s1, s2, _ = shrink_step_adaptive!(A, y, w, r, λ, s1, s2; thresh=1_000)
        resize!(A, newlen)
        λ_new = (s1 - r) / s2
        
        if abs(λ_new - λ) ≤ tol * (one(T) + abs(λ))
            λ = λ_new
            break
        end
        
        if extraploop
            eval_hat = λ * s2 - s1 + r
            lenA = length(A)
            lenA_safe = length(A_safe)
            
            if eval_hat > 0
                A = A_safe; lenA = lenA_safe; s1 = s1_safe; 
                s2 = s2_safe; λ_new = λ_safe; extrap = false
            else
                resize!(A_safe, lenA)
                copyto!(A_safe, 1, A, 1, lenA)
                s1_safe = s1; s2_safe = s2; λ_safe = λ_new
            end
        end
        
        if extrap
            λ_hat = λ + extrap_factor * (λ_new - λ)
            λ = max(λ_hat, zero(T))
        else
            λ = λ_new
        end
        
    end

    # --- write solution: x[i] = y[i] - λ w[i] on active set; others 0
    value_list = Float64[]
    @inbounds for j in A
        push!(value_list, y[j] - w[j]*λ)
    end
    x = sparsevec(A, value_list, length(y))
    
    return x, λ, iters, sum_A
end



"""
    wfilter(data, w, b)

Filter technique for weighted simplex projection
"""
function wfilter(data::Array{Float64, 1}, w::Array{Float64, 1}, b::Real)
   let
        #initialize
        active_list = Int[1]
        s1 = data[1] * w[1]
        s2 = w[1]^2
        pivot = (s1 - b)/(s2)
        wait_list = Int[]
        #check all terms
        for i in 2:length(data)
            #remove inactive terms
            if data[i]/w[i] > pivot
                #update pivot
                pivot = (s1 + data[i] * w[i] - b)/(s2 + w[i]^2)
                if pivot > (data[i] * w[i] - b)/(w[i]^2)
                    push!(active_list, i)
                    s1 += data[i] * w[i]
                    s2 += w[i]^2
                else
                    #for large pivot
                    append!!(wait_list, active_list)
                    active_list = Int[i]
                    s1 = data[i] * w[i]
                    s2 = w[i]^2
                    pivot = (s1 - b)/s2
                end
            end
        end
        #reuse terms from waiting list
        for j in wait_list
            if data[j]/w[j] >pivot
                push!(active_list, j)
                s1 += data[j]*w[j]
                s2 += w[j]^2
                pivot = (s1 - b)/s2
            end
        end
        return active_list, s1, s2
    end
end

"""
    wcheckL(active_list, s1, s2, data, w, b)

Remaining algorithm (after Filter) of weighted simplex projection based on Condat's method
"""
function wcheckL(active_list::Array{Int, 1}, s1::Float64, s2::Float64, data::Array{Float64, 1}, w::Array{Float64, 1}, b::Float64)
    let
        pivot = (s1 - b)/s2
        
        iters = 0
        while true
            iters += 1
            length_cache = length(active_list)
            for _ in 1:length_cache
                i = popfirst!(active_list)
                if data[i]/w[i] > pivot
                    push!(active_list, i)
                else
                    s1 = s1 - data[i]*w[i]
                    s2 = s2 - w[i]^2
                    pivot = (s1 - b)/s2
                end
            end
            
            if length_cache == length(active_list)
                break
            end
        end

        value_list = Float64[]
        for j in active_list
            push!(value_list, data[j] - w[j]*pivot)
        end
        
        x = sparsevec(active_list, value_list, length(data))
        
        return x, pivot, iters
    end
end

"""
    parallel_wfilter(data, w, b, numthread)

Parallel filter technique for weighted simplex projection
"""
function parallel_wfilter(data::Array{Float64, 1}, w::Array{Float64, 1}, b::Real, numthread::Int)
    #the length for subvectors
    width = floor(Int, length(data)/numthread)
    #lock global value
    spl = SpinLock()
    #initialize a global list
    glist = Int[]
    gs1 = 0.0
    gs2 = 0.0
    @threads for id in 1:numthread
        let
            #determine start and end position for subvectors
            local st = (id-1) * width + 1
            if id == numthread
                local en = length(data)
            else
                local en = id * width
            end
            local active_list = Int[st]
            local s1 = data[st] * w[st]
            local s2 = w[st]^2
            local pivot = (s1 - b)/(s2)
            local wait_list = Int[]
            #check all terms
            for i in (st+1):en
                #remove inactive terms
                if data[i]/w[i] > pivot
                    #update pivot
                    
                    wyi = data[i] * w[i]
                    w2i = w[i]^2
                    
                    pivot = (s1 + wyi - b)/(s2 + w2i)
                    if pivot > (wyi - b)/w2i
                        push!(active_list, i)
                        s1 += wyi
                        s2 += w2i
                    else
                        #for large pivot
                        append!!(wait_list, active_list)
                        active_list = Int[i]
                        s1 = wyi
                        s2 = w2i
                        pivot = (s1 - b)/s2
                    end
                    
                end
            end
            #reuse terms from waiting list
            for j in wait_list
                if data[j]/w[j] >pivot
                    push!(active_list, j)
                    s1 += data[j]*w[j]
                    s2 += w[j]^2
                    pivot = (s1 - b)/s2
                end
            end
            while true
                length_cache = length(active_list)
                for _ in 1:length_cache
                    i = popfirst!(active_list)
                    if data[i]/w[i] > pivot
                        push!(active_list, i)
                    else
                        s1 = s1 - data[i]*w[i]
                        s2 = s2 - w[i]^2
                        pivot = (s1 - b)/s2
                    end
                end
                if length_cache == length(active_list)
                    break
                end
            end
            #reduce with locking
            lock(spl)
            append!!(glist, active_list)
            gs1 += s1
            gs2 += s2
            unlock(spl)
        end
    end
    return glist, gs1, gs2
end

"""
    wcondat_s(data, w, b)

Weighted simplex projection based on serial Condat's method
"""
function wcondat_s(data::Array{Float64, 1}, w::Array{Float64, 1}, b::Real)::AbstractVector
    active_list, s1, s2 = wfilter(data, w, b)
    x, pivot, iters = wcheckL(active_list, s1, s2, data, w, b)
    
    return x, pivot, iters
end

"""
    wcondat_s(data, w, b, numthread)

Weighted simplex projection based on parallel Condat's method
"""
function wcondat_p(data::Array{Float64, 1}, w::Array{Float64, 1}, b::Real, numthread)
    active_list, s1, s2 = parallel_wfilter(data, w, b, numthread)
    
    x, pivot, iters = wcheckL(active_list, s1, s2, data, w, b)
    
    
    return x, pivot, iters 
end


end
