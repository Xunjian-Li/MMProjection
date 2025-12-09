module UtilityFunctions

using Base.Threads
using BangBang
using SparseArrays

using LinearAlgebra, ProximalOperators
import ProximalOperators: prox, prox!


export make_set, build_sets, Loss, prox, make_projector, IndWSimplex, pav_nondecreasing!, pav_nonincreasing!
export CenteredBallL2, CenteredSphereL2


@inline _iszerovec(c; atol::Real=0.0, rtol::Real=1e-12) = norm(c) ≤ max(atol, rtol)

# ============================================================
# 1. Monotone decreasing cone:
#    C = { x ∈ ℝ^p : x₁ ≥ x₂ ≥ ⋯ ≥ x_p ≥ 0 }
# ============================================================

struct IndMonoDec
    p::Int   # dimension
end

"""
PAV projection onto the nondecreasing cone:

    x₁ ≤ x₂ ≤ ⋯ ≤ xₙ

Works in-place on `y`.
"""
function pav_nondecreasing!(y::AbstractVector{T}) where {T<:Real}
    n = length(y)
    g = similar(y)
    w = similar(y)
    k = 0

    for i in 1:n
        k += 1
        g[k] = y[i]
        w[k] = one(T)
        while k > 1 && g[k-1] > g[k]
            s = g[k-1]*w[k-1] + g[k]*w[k]
            w[k-1] += w[k]
            g[k-1] = s / w[k-1]
            k -= 1
        end
    end

    idx = 1
    for j in 1:k
        v   = g[j]
        cnt = Int(round(w[j]))
        @inbounds for _ in 1:cnt
            y[idx] = v
            idx += 1
        end
    end
    return y
end

"""
PAV projection onto the nonincreasing cone:

    x₁ ≥ x₂ ≥ ⋯ ≥ xₙ

Implemented by negating, using the nondecreasing PAV,
and negating back.
"""
function pav_nonincreasing!(y::AbstractVector{T}) where {T<:Real}
    @. y = -y
    pav_nondecreasing!(y)
    @. y = -y
    return y
end

"""
prox! for IndMonoDec:

    dst ← proj_{x₁≥⋯≥x_p}(x)

If you also want x_p ≥ 0, you can add a `max(dst,0)` at the end.
"""
function prox!(dst::AbstractVector{T}, C::IndMonoDec,
               x::AbstractVector{T}, γ::Real) where {T<:Real}
    @assert length(dst) == length(x) == C.p
    copyto!(dst, x)
    pav_nonincreasing!(dst)
    # If you want the extra constraint x ≥ 0, uncomment:
    # @. dst = max(dst, zero(T))
    return dst
end

function prox(C::IndMonoDec, x::AbstractVector{T}, γ::Real) where {T<:Real}
    @assert length(x) == C.p
    dst = similar(x)
    prox!(dst, C, x, γ)
    return dst, 0.0
end

# ============================================================
# 2. Centered L2 ball: B₂(c,r) = { x : ‖x-c‖₂ ≤ r }
# ============================================================

struct CenteredBallL2{T,VT}
    r::T
    c::VT
    inner   # ProximalOperators.IndBallL2
end

CenteredBallL2(r::Real, c::AbstractVector) =
    CenteredBallL2(r, collect(c), ProximalOperators.IndBallL2(r))

"""
prox! for centered L2 ball:

    dst ← proj_{‖·-c‖≤r}(x)
"""
function prox!(dst::AbstractVector, C::CenteredBallL2, x::AbstractVector, γ::Real)
    z = x .- C.c
    prox!(dst, C.inner, z, γ)
    @. dst = dst + C.c
    return dst
end

function prox(C::CenteredBallL2, x::AbstractVector, γ::Real)
    y = similar(x)
    prox!(y, C, x, γ)
    return y, 0.0
end

# ============================================================
# 3. Centered L2 sphere: { x : ‖x-c‖₂ = r }
# ============================================================

struct CenteredSphereL2{T,VT}
    r::T
    c::VT
    inner   # ProximalOperators.IndSphereL2
end

CenteredSphereL2(r::Real, c::AbstractVector) =
    CenteredSphereL2(r, collect(c), ProximalOperators.IndSphereL2(r))

"""
prox! for centered L2 sphere:

    dst ← proj_{‖·-c‖=r}(x)
"""
function prox!(dst::AbstractVector, C::CenteredSphereL2, x::AbstractVector, γ::Real)
    @assert length(C.c) == length(x) "sphere center and x must have same length"
    z = x .- C.c  
    prox!(dst, C.inner, z, γ)
    @. dst = dst + C.c 
    return dst
end

function prox(C::CenteredSphereL2, x::AbstractVector, γ::Real)
    y = similar(x)
    prox!(y, C, x, γ)
    return y, 0.0
end

# ============================================================
# 4. Weighted simplex:
#    C = { x ≥ 0 : wᵀ x = b }
# ============================================================

"""
Indicator of a weighted simplex:

    C = { x ∈ ℝ^p : x ≥ 0, wᵀ x = b },

where w ∈ ℝ^p (weights), b ∈ ℝ.
"""
struct IndWSimplex{T,VT<:AbstractVector{T}}
    w::VT      # weights
    b::T       # right-hand side
end

"""
Outer constructor that normalizes the storage:

- `w` is copied to a dense Vector{T}
- `b` is converted to type `T`
"""
IndWSimplex(w::AbstractVector{T}, b::Real) where {T<:Real} =
    IndWSimplex{T,Vector{T}}(collect(w), T(b))


"""
    wfilter(data, w, b)

Filter step for weighted simplex projection.

Works for any `T<:Real` (including ForwardDiff.Dual).
"""
function wfilter(data::AbstractVector{T},
                 w::AbstractVector{T},
                 b::T) where {T<:Real}

    @assert length(data) == length(w)

    # Initialize
    active_list = Int[1]
    s1 = data[1] * w[1]      # type T
    s2 = w[1]^2              # type T
    pivot = (s1 - b) / s2    # type T

    wait_list = Int[]

    # Scan all terms
    for i in 2:length(data)
        # Only consider coordinates where data[i]/w[i] > pivot
        if data[i] / w[i] > pivot
            # Update pivot with i tentatively added
            s1_new = s1 + data[i] * w[i]
            s2_new = s2 + w[i]^2
            pivot_new = (s1_new - b) / s2_new

            if pivot_new > (data[i] * w[i] - b) / (w[i]^2)
                # i stays in active set
                push!(active_list, i)
                s1 = s1_new
                s2 = s2_new
                pivot = pivot_new
            else
                # pivot becomes large => restart active set from i
                append!!(wait_list, active_list)
                active_list = Int[i]
                s1 = data[i] * w[i]
                s2 = w[i]^2
                pivot = (s1 - b) / s2
            end
        end
    end

    # Reuse indices from waiting list
    for j in wait_list
        if data[j] / w[j] > pivot
            push!(active_list, j)
            s1 += data[j] * w[j]
            s2 += w[j]^2
            pivot = (s1 - b) / s2
        end
    end

    return active_list, s1, s2
end


"""
    wcheckL(active_list, s1, s2, data, w, b)

Final step of Condat's weighted simplex projection, after `wfilter`.

All arguments are generic in `T<:Real` so this works with Dual too.
Returns a `SparseVector{T}` of length `length(data)`.
"""
function wcheckL(active_list::Vector{Int},
                 s1::T,
                 s2::T,
                 data::AbstractVector{T},
                 w::AbstractVector{T},
                 b::T) where {T<:Real}

    pivot = (s1 - b) / s2    # T

    # Iteratively remove indices that violate the KKT conditions
    while true
        length_cache = length(active_list)

        for _ in 1:length_cache
            i = popfirst!(active_list)
            if data[i] / w[i] > pivot
                # keep i
                push!(active_list, i)
            else
                # drop i and update sums and pivot
                s1 -= data[i] * w[i]
                s2 -= w[i]^2
                pivot = (s1 - b) / s2
            end
        end

        # If no index was removed, we are done
        if length_cache == length(active_list)
            break
        end
    end

    # Build the sparse projection vector
    values = T[]
    for j in active_list
        push!(values, data[j] - w[j] * pivot)
    end

    # SparseVector{T,Int}
    return sparsevec(active_list, values, length(data))
end


# -------- Generic weighted simplex projection --------------------------
# Must be generic in T<:Number to support Dual for AD.

"""
Weighted simplex projection (Condat-like):

    wcondat_s(data, w, b)

Projects `data` onto:

    { x ≥ 0 : wᵀ x = b }.

All arguments share the same element type `T<:Number`.
"""
function wcondat_s(data::AbstractVector{T},
                   w::AbstractVector{T},
                   b::T) where {T<:Number}
    active_list, s1, s2 = wfilter(data, w, b)
    return wcheckL(active_list, s1, s2, data, w, b)
end

"""
In-place prox for IndWSimplex:

    dst ← proj_{x≥0, wᵀx=b}(x)

We promote stored weights and rhs to the same element type as `x`
(works with ForwardDiff.Dual as long as wfilter/wcheckL are generic).
"""
function prox!(dst::AbstractVector{T}, C::IndWSimplex{T,VT},
               x::AbstractVector{T}, γ::Real) where {T<:Number, VT<:AbstractVector{T}}
    @assert length(dst) == length(x) == length(C.w)
    y = wcondat_s(x, C.w, C.b)
    copyto!(dst, y)
    return dst
end

"""
Out-of-place prox for IndWSimplex.
"""
function prox(C::IndWSimplex{T,VT}, x::AbstractVector{T2}, γ::Real) where
        {T<:Number, VT<:AbstractVector{T}, T2<:Number}
    # allow x to have element type T2 (e.g., Dual) and promote w,b to T2
    wT = T2.(C.w)
    bT = T2(C.b)
    y  = wcondat_s(T2.(x), wT, bT)
    return y, 0.0
end

# ============================================================
# 5. Factory functions: make_set / build_sets
# ============================================================

"""
Read radius from a spec NamedTuple.

Accepted keys:
  :r, :ρ, :radius, or (legacy) scalar :c.
"""
_get_radius(spec) = haskey(spec, :r)      ? spec.r      :
                    haskey(spec, :ρ)      ? spec.ρ      :
                    haskey(spec, :radius) ? spec.radius :
                    (haskey(spec, :c) && spec.c isa Real ? spec.c :
                      throw(ArgumentError(":sphereL2 needs :r (or :ρ/:radius); legacy :c accepted only if Real")))

"""
Construct a proximal set/function from a NamedTuple specification.

Supported `spec.type`:
  :ballL2    -> (centered or origin L2 ball)
  :simplex   -> standard simplex
  :wsimplex  -> weighted simplex {x ≥ 0 : wᵀ x = b}
  :monoDec   -> monotone decreasing cone
  :affine    -> affine subspace
  :halfspace -> halfspace
  :box       -> axis-aligned box
  :sphereL2  -> L2 sphere (centered or origin)
  :psd       -> positive semidefinite cone (matrix domain)
"""

function make_set(spec::NamedTuple; p::Union{Nothing,Int}=nothing)
    @assert haskey(spec, :type) "spec must have :type key"
    kind = spec.type

    if kind === :ballL2
        @assert haskey(spec, :r) || haskey(spec, :ρ) ":ballL2 needs :r (or :ρ)"
        r = haskey(spec, :r) ? spec.r : spec.ρ

        if haskey(spec, :c)
            cvec = collect(spec.c)
            if _iszerovec(cvec)
                return ProximalOperators.IndBallL2(r)
            else
                return CenteredBallL2(r, cvec)
            end
        else
            return ProximalOperators.IndBallL2(r)
        end

    elseif kind === :simplex
        @assert haskey(spec, :c)
        return IndSimplex(spec.c)

    elseif kind === :wsimplex
        @assert haskey(spec, :w) && haskey(spec, :b) ":wsimplex needs :w (weights) and :b (rhs)"
        w = collect(spec.w)
        b = spec.b
        return IndWSimplex(w, b)

    elseif kind === :monoDec
        # C = {x ∈ ℝ^p : x₁ ≥ x₂ ≥ ⋯ ≥ x_p ≥ 0}
        p_eff = haskey(spec, :p) ? spec.p : p
        @assert p_eff !== nothing ":monoDec needs dimension p (either in spec or argument)"
        return IndMonoDec(p_eff)

    elseif kind === :affine
        @assert haskey(spec, :α) && haskey(spec, :c)
        return IndAffine(spec.α, spec.c)

    elseif kind === :halfspace
        @assert haskey(spec, :α) && haskey(spec, :c)
        return IndHalfspace(spec.α, spec.c)

    elseif kind === :box
        lo_given = haskey(spec, :lo_vec); hi_given = haskey(spec, :hi_vec)
        if lo_given && hi_given
            return IndBox(spec.lo_vec, spec.hi_vec)
        else
            @assert haskey(spec, :low) && haskey(spec, :high)
            p_eff = haskey(spec, :p) ? spec.p : p
            @assert p_eff !== nothing "box with scalar bounds needs dimension p"
            return IndBox(fill(spec.low, p_eff), fill(spec.high, p_eff))
        end
            
    elseif kind === :psd
        return ProximalOperators.IndPSD()

    elseif kind === :sphereL2
        r = _get_radius(spec)
        if haskey(spec, :center)
            cvec = collect(spec.center)
            if p !== nothing
                @assert length(cvec) == p "sphereL2 center length $(length(cvec)) ≠ p=$p"
            end
            if _iszerovec(cvec)
                return ProximalOperators.IndSphereL2(r)
            else
                return CenteredSphereL2(r, cvec)
            end
        else
            return ProximalOperators.IndSphereL2(r)
        end
            

    else
        error("unknown spec type: $kind")
    end
end

"""
Build a list of proximal sets from a list of NamedTuple specs.

Example:
    specs = [
        (type = :wsimplex, w = w, b = 1.0),
        (type = :monoDec,  p = length(x)),
    ]
    sets = build_sets(specs; p = length(x))
"""
build_sets(specs::Vector{<:NamedTuple}; p::Union{Nothing,Int}=nothing) =
    Any[ make_set(s; p=p) for s in specs ]
    

make_projector(C) = (u -> prox(C, u)[1])
    
    
function Loss(sets, x_project)
    r = length(sets)
    dist = 0.0
    for i in 1:r
        dist += sum(abs.(prox(sets[i], x_project)[1] .- x_project))
    end
    return dist
end
    
    
end