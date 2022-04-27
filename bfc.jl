using LinearAlgebra
using SparseArrays
using PyPlot

const USE_GPU = false
using ParallelStencil
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2);
else
    @init_parallel_stencil(Threads, Float64, 2);
end

function NS_solve()
    nₓ, nᵧ , nₜ = 60, 60, 10000

    w, h = 1.0, 1.0
    dξ, dη = w/nₓ, h/nᵧ
    dt = min(dξ^2, dη^2)
    ρ = 1.0
    ν = 0.01

    uₓ_1    = @zeros(nₓ, nᵧ)
    uᵧ_1    = @zeros(nₓ, nᵧ)
    uₓ_2    = @zeros(nₓ, nᵧ)
    uᵧ_2    = @zeros(nₓ, nᵧ)

    u¹_1    = @zeros(nₓ, nᵧ)
    u²_1    = @zeros(nₓ, nᵧ)
    u¹_2    = @zeros(nₓ, nᵧ)
    u²_2    = @zeros(nₓ, nᵧ)

    u★ₓ_1    = @zeros(nₓ, nᵧ)
    u★ᵧ_1    = @zeros(nₓ, nᵧ)
    u★ₓ_2    = @zeros(nₓ, nᵧ)
    u★ᵧ_2    = @zeros(nₓ, nᵧ)

    u★¹_1    = @zeros(nₓ, nᵧ)
    u★²_1    = @zeros(nₓ, nᵧ)
    u★¹_2    = @zeros(nₓ, nᵧ)
    u★²_2    = @zeros(nₓ, nᵧ)    

    p       = @zeros(nₓ, nᵧ)
    p˒ξ     = @zeros(nₓ, nᵧ)
    p˒η     = @zeros(nₓ, nᵧ)
    
    div_u★  = @zeros(nₓ, nᵧ)

    x       = @zeros(2nₓ+1, 2nᵧ+1)
    y       = @zeros(2nₓ+1, 2nᵧ+1)

    x˒ξ_1   = @zeros(nₓ, nᵧ)
    x˒η_1   = @zeros(nₓ, nᵧ)
    x˒ξ_2   = @zeros(nₓ, nᵧ)
    x˒η_2   = @zeros(nₓ, nᵧ)

    y˒ξ_1   = @zeros(nₓ, nᵧ)
    y˒η_1   = @zeros(nₓ, nᵧ)
    y˒ξ_2   = @zeros(nₓ, nᵧ)
    y˒η_2   = @zeros(nₓ, nᵧ)

    J_1     = @zeros(nₓ, nᵧ)
    J_2     = @zeros(nₓ, nᵧ)

    inv_J_1 = @zeros(nₓ, nᵧ)
    inv_J_2 = @zeros(nₓ, nᵧ)    

    g¹ₓ_1   = @zeros(nₓ, nᵧ)
    g¹ᵧ_1   = @zeros(nₓ, nᵧ)
    g²ₓ_1   = @zeros(nₓ, nᵧ)
    g²ᵧ_1   = @zeros(nₓ, nᵧ)

    g¹ₓ_2   = @zeros(nₓ, nᵧ)
    g¹ᵧ_2   = @zeros(nₓ, nᵧ)
    g²ₓ_2   = @zeros(nₓ, nᵧ)
    g²ᵧ_2   = @zeros(nₓ, nᵧ)                
    
    g₁₁_1   = @ones(nₓ, nᵧ)
    g₁₂_1   = @zeros(nₓ, nᵧ)
    g₂₂_1   = @ones(nₓ, nᵧ)

    g₁₁_2   = @ones(nₓ, nᵧ)
    g₁₂_2   = @zeros(nₓ, nᵧ)
    g₂₂_2   = @ones(nₓ, nᵧ)
    
    g¹¹_1   = @ones(nₓ, nᵧ)
    g¹²_1   = @zeros(nₓ, nᵧ)
    g²²_1   = @ones(nₓ, nᵧ)

    g¹¹_2   = @ones(nₓ, nᵧ)
    g¹²_2   = @zeros(nₓ, nᵧ)
    g²²_2   = @ones(nₓ, nᵧ)

    x .= repeat((0:(2nₓ))*w/(2nₓ), 1, 2nᵧ+1)'
    y .= h/w*x'
    
    @. x˒ξ_1[2:end-1,2:end-1] = (x[4:2:end-3,4:2:end-3] - x[2:2:end-5,4:2:end-3])/dξ
    @. y˒ξ_1[2:end-1,2:end-1] = (y[4:2:end-3,4:2:end-3] - y[2:2:end-5,4:2:end-3])/dξ

    @. x˒ξ_2[2:end-1,2:end-1] = (x[5:2:end-2,3:2:end-4] - x[3:2:end-4,3:2:end-4])/dξ
    @. y˒ξ_2[2:end-1,2:end-1] = (y[5:2:end-2,3:2:end-4] - y[3:2:end-4,3:2:end-4])/dξ

    @. x˒η_1[2:end-1,2:end-1] = (x[3:2:end-4,5:2:end-2] - x[3:2:end-4,3:2:end-4])/dη
    @. y˒η_1[2:end-1,2:end-1] = (y[3:2:end-4,5:2:end-2] - y[3:2:end-4,3:2:end-4])/dη    

    @. x˒η_2[2:end-1,2:end-1] = (x[4:2:end-3,5:2:end-2] - x[4:2:end-3,3:2:end-4])/dη
    @. y˒η_2[2:end-1,2:end-1] = (y[4:2:end-3,5:2:end-2] - y[4:2:end-3,3:2:end-4])/dη    
    
    return x, y, x˒ξ_1, x˒η_1, y˒ξ_1, y˒η_1
end

x, y, x_ξ_1, x_η_1, y_ξ_1, y_η_1 = NS_solve()
