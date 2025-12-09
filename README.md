# **MMProjection.jl**

A Julia package for fast and scalable **MM-based projection algorithms**.

---

## 📌 **Overview**

**MMProjection.jl** implements a collection of **majorization–minimization (MM)** projection algorithms that efficiently compute projections onto various constraint sets, including:

- Simplex  
- ℓ₂ balls  
- Hyperplanes  
- Intersection of convex sets  
- Custom constraints via modular operators  

This package focuses on **speed**, **numerical stability**, and **scalability**, with support for GPU acceleration and large-dimensional problems.

A full explanation of the algorithms and theory is available on the project website:

🔗 **[Documentation & Usage Guide](<在这里放你说的网页链接>)**

---

## 🚀 **Installation**

Install the package via Julia's package manager:

```julia
using Pkg
Pkg.add(url="https://github.com/Xunjian-Li/MMProjection.jl")
```


## 🔧 **Basic Usage

Projection onto the intersection of multiple constraints:

```julia
using MMProjection

Random.seed!(2025)

m, ρ, σ = 6, 0.95, 1.0
R = fill(ρ, m, m); @inbounds for i in 1:m R[i,i] = 1.0 end
Σ = (σ^2) .* R
dist = MvNormal(zeros(m), Σ)

n = 1000
samples = rand(dist, n)
c_list  = [vec(samples[i, :]) for i in 1:m]
A = hcat(c_list...)
b = randn(m)
x = rand(n)

specs = [
    (type=:halfspace, α=A[:,1], c=b[1]),
    (type=:halfspace, α=A[:,2], c=b[2]),
    (type=:halfspace, α=A[:,3], c=b[3]),
    (type=:halfspace, α=A[:,4], c=b[4]),
    (type=:halfspace, α=A[:,5], c=b[5]),
    (type=:halfspace, α=A[:,6], c=b[6]),
    (type=:simplex,   c=1.0),
    # (type=:ballL2,   ρ=1.0),
    # (type=:ballL1,   ρ=1.2),
    # (type=:ballLinf, ρ=2.0),
    # (type=:affine,   α=αvec, c=c0),
    # (type=:box,      low=0.0, high=Inf, p=length(x0)),
    # (type=:box,      lo_vec=lo, hi_vec=hi),
]

r = length(specs)
rLambda = zeros(n, r);
x0 = similar(x);

sets = build_sets(specs; p=length(specs));

C = sets[end]

kwargs2 = (
    A=A, 
    b=b, 
    mode=:halfspaces,
    tol=1e-10, 
    max_outer=100
)


kwargs = (
#     variant  = :AAI,   # :AAI or :AAII
    m        = 4,      # memory
    tol      = 1e-10,   # stopping tolerance
    τ        = 1.0,    # damping
    # optional extras if your runner accepts them:
    maxiter  = 10000,
    λ        = 1e-10,
    cnd_max  = 1e8,
)

xN2, λ2, iters2, state2 = project_halfspaces_active_set(C, x; kwargs2...)

x_AAI, Λ_AAI, iters_AAI, ok_AAI = run_dykstra!(copy(x0), copy(x), copy(rLambda), sets; AAvariant  = :AAI, kwargs...)

x_AAII, Λ_AAII, iters_AAII, ok_AAII = run_dykstra!(copy(x0), copy(x), copy(rLambda), sets; AAvariant  = :AAII, kwargs...)

x_Dy, Λ_Dy, iters_Dy, ok_Dy = run_dykstra!(copy(x0), copy(x), copy(rLambda), sets; AAvariant  = :none, kwargs...)


```

## 🧪 **Testing

To run tests locally:

```julia
using Pkg
Pkg.test()
```







