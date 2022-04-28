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

macro dξ_1(A, i, j)
    return :(($(esc(A))[$(esc(i))+1,$(esc(j))]-$(esc(A))[$(esc(i))-1,$(esc(j))])/2)
end

macro dξ_2(A, i, j)
    return :(($(esc(A))[$(esc(i))+1,$(esc(j))]-$(esc(A))[$(esc(i))-1,$(esc(j))])/2)
end

macro dη_1(A, i, j)
    return :(($(esc(A))[$(esc(i)),$(esc(j))+1]-$(esc(A))[$(esc(i)),$(esc(j))-1])/2)
end

macro dη_2(A, i, j)
    return :(($(esc(A))[$(esc(i)),$(esc(j))+1]-$(esc(A))[$(esc(i)),$(esc(j))-1])/2)
end

function compute_u★!(nₓ,
                     nᵧ,
                     dt,
                     dξ,
                     dη,
                     ν,
                     J_1,
                     J_2,
                     inv_J_1,
                     inv_J_2,
                     g¹¹_1,
                     g¹²_1,
                     g²¹_1,
                     g²²_1,
                     g¹¹_2,
                     g¹²_2,
                     g²¹_2,
                     g²²_2,
                     g¹ₓ_1,
                     g¹ᵧ_1,
                     g²ₓ_2,
                     g²ᵧ_2,
                     uₓ_1,
                     uᵧ_1,
                     uₓ_2,
                     uᵧ_2,
                     u¹_1,
                     u²_1,
                     u¹_2,
                     u²_2,
                     u★¹_1,
                     u★²_2)


    A1_1 = @zeros(nₓ, nᵧ)
    A2_1 = @zeros(nₓ, nᵧ)
    A1_2 = @zeros(nₓ, nᵧ)
    A2_2 = @zeros(nₓ, nᵧ)    

    B11 = @zeros(nₓ, nᵧ)
    B12 = @zeros(nₓ, nᵧ)
    B21 = @zeros(nₓ, nᵧ)
    B22 = @zeros(nₓ, nᵧ)        

    for i∈2:nₓ-1, j∈2:nᵧ-1
        B11[i,j] = inv_J_1[i,j]*(g¹¹_1[i,j]*@dξ_1(uₓ_1,i,j)/dξ + g¹²_1[i,j]*@dη_1(uₓ_1,i,j)/dη)
        B12[i,j] = inv_J_1[i,j]*(g²¹_1[i,j]*@dξ_1(uₓ_1,i,j)/dξ + g²²_1[i,j]*@dη_1(uₓ_1,i,j)/dη)
        
        B21[i,j] = inv_J_2[i,j]*(g¹¹_2[i,j]*@dξ_2(uᵧ_2,i,j)/dξ + g¹²_2[i,j]*@dη_2(uᵧ_2,i,j)/dη)
        B22[i,j] = inv_J_2[i,j]*(g²¹_2[i,j]*@dξ_2(uᵧ_2,i,j)/dξ + g²²_2[i,j]*@dη_2(uᵧ_2,i,j)/dη)        
    end
    

    for i∈2:nₓ-1, j∈2:nᵧ-1
        A1_1[i,j] = dt*(-(u¹_1[i,j]*@dξ_1(uₓ_1,i,j)/dξ + u²_1[i,j]*@dη_1(uₓ_1,i,j)/dη) + ν*J_1[i,j]*(@dξ_1(B11,i,j)/dξ + @dη_1(B12,i,j)/dη))
        A2_2[i,j] = dt*(-(u¹_2[i,j]*@dξ_1(uᵧ_2,i,j)/dξ + u²_2[i,j]*@dη_2(uᵧ_2,i,j)/dη) + ν*J_2[i,j]*(@dξ_2(B21,i,j)/dξ + @dη_2(B22,i,j)/dη))
    end

    A1_2[2:end-1,2:end-1] = 1/4*(A1_1[2:end-1,1:end-2] + A1_1[3:end,1:end-2] + A1_1[2:end-1,2:end-1] + A1_1[3:end,2:end-1])
    A2_1[2:end-1,2:end-1] = 1/4*(A2_2[1:end-2,2:end-1] + A2_2[1:end-2,3:end] + A2_2[2:end-1,2:end-1] + A2_2[2:end-1,3:end])

    # boundary conditions on A1, A2 ?

    for i∈2:nₓ-1, j∈2:nᵧ-1
        u★¹_1[i,j] = A1_1[i,j]*g¹ₓ_1[i,j] + A2_1[i,j]*g¹ᵧ_1[i,j]
        u★²_2[i,j] = A1_2[i,j]*g²ₓ_2[i,j] + A2_2[i,j]*g²ᵧ_2[i,j]
    end

    # u★¹_2, u★²_1 not needed to compute the divergence !!!
end

function compute_div_u★!(nₓ,
                        nᵧ,
                        dt,
                        dξ,
                        dη,
                        ρ,
                        J_1,
                        J_2,
                        inv_J_1,
                        inv_J_2,
                        u★¹_1,
                        u★²_2,
                        div_u★)

    for i∈2:nₓ-1, j∈2:nᵧ-1
        div_u★[i,j] = (ρ/dt)*(1/4)*(J_1[i,j]+J_1[i+1,j]+J_2[i,j]+J_2[i,j+1])*((u★¹_1[i+1,j]*inv_J_1[i+1,j]-u★¹_1[i,j]*inv_J_1[i,j])/dξ + (u★²_2[i,j+1]*inv_J_2[i,j+1]-u★²_2[i,j]*inv_J_2[i,j])/dη)
    end
end

function update_u!(nₓ,
                   nᵧ,
                   dt,
                   dξ,
                   dη,
                   ρ,
                   g¹¹_1,
                   g¹²_1,
                   g²¹_1,
                   g²²_1,
                   g¹¹_2,
                   g¹²_2,
                   g²¹_2,
                   g²²_2,
                   g₁ₓ_1,
                   g₁ᵧ_1,
                   g₁ₓ_2,
                   g₁ᵧ_2,
                   g₂ₓ_1,
                   g₂ᵧ_1,                   
                   g₂ₓ_2,
                   g₂ᵧ_2,
                   u★¹_1,
                   u★²_2,
                   p,
                   u¹_1,
                   u²_1,
                   u¹_2,
                   u²_2,
                   uₓ_1,
                   uᵧ_1,
                   uₓ_2,
                   uᵧ_2)

    for i∈2:nₓ-1, j∈2:nᵧ-1
        u¹_1[i,j] = u★¹_1[i,j] - (dt/ρ)*(g¹¹_1[i,j]*(p[i,j]-p[i-1,j])/dξ + g¹²_1[i,j]*((p[i,j+1]+p[i-1,j+1])/2-(p[i,j-1]+p[i-1,j-1])/2)/(2dη))
        u²_2[i,j] = u★²_2[i,j] - (dt/ρ)*(g²¹_2[i,j]*((p[i+1,j]-p[i+1,j-1])/2-(p[i-1,j]-p[i-1,j-1])/2)/(2dξ) + g²²_2[i,j]*(p[i,j]-p[i,j-1])/dη)
    end

    u¹_2[2:end-1,2:end-1] = 1/4*(u¹_1[2:end-1,1:end-2] + u¹_1[3:end,1:end-2] + u¹_1[2:end-1,2:end-1] + u¹_1[3:end,2:end-1])
    u²_1[2:end-1,2:end-1] = 1/4*(u²_2[1:end-2,2:end-1] + u²_2[1:end-2,3:end] + u²_2[2:end-1,2:end-1] + u²_2[2:end-1,3:end])    


    @. uₓ_1 = u¹_1*g₁ₓ_1 + u²_1*g₂ₓ_1
    @. uᵧ_2 = u¹_2*g₁ᵧ_2 + u²_2*g₂ᵧ_2

    uₓ_2[2:end-1,2:end-1] = 1/4*(uₓ_1[2:end-1,1:end-2] + uₓ_1[3:end,1:end-2] + uₓ_1[2:end-1,2:end-1] + uₓ_1[3:end,2:end-1])
    uᵧ_1[2:end-1,2:end-1] = 1/4*(uᵧ_2[1:end-2,2:end-1] + uᵧ_2[1:end-2,3:end] + uᵧ_2[2:end-1,2:end-1] + uᵧ_2[2:end-1,3:end])        
    
    # boundary conditions !!!
    
end
                   
                        

function NS_solve()
    nₓ, nᵧ , nₜ = 50, 50, 10

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

    u★¹_1    = @zeros(nₓ, nᵧ)
    u★²_2    = @zeros(nₓ, nᵧ)    

    p       = @zeros(nₓ, nᵧ)
    p˒ξ     = @zeros(nₓ, nᵧ)
    p˒η     = @zeros(nₓ, nᵧ)
    
    div_u★  = @zeros(nₓ, nᵧ)

    x       = @zeros(2nₓ+1, 2nᵧ+1)
    y       = @zeros(2nₓ+1, 2nᵧ+1)

    x˒ξ_1   = @ones(nₓ, nᵧ)
    x˒η_1   = @zeros(nₓ, nᵧ)
    x˒ξ_2   = @ones(nₓ, nᵧ)
    x˒η_2   = @zeros(nₓ, nᵧ)

    y˒ξ_1   = @zeros(nₓ, nᵧ)
    y˒η_1   = @ones(nₓ, nᵧ)
    y˒ξ_2   = @zeros(nₓ, nᵧ)
    y˒η_2   = @ones(nₓ, nᵧ)

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

    g₁ₓ_1   = @zeros(nₓ, nᵧ)
    g₁ᵧ_1   = @zeros(nₓ, nᵧ)
    g₂ₓ_2   = @zeros(nₓ, nᵧ)
    g₂ᵧ_2   = @zeros(nₓ, nᵧ)
    
    g¹¹_1   = @ones(nₓ, nᵧ)
    g¹²_1   = @zeros(nₓ, nᵧ)
    g²¹_1   = @zeros(nₓ, nᵧ)
    g²²_1   = @ones(nₓ, nᵧ)

    g¹¹_2   = @ones(nₓ, nᵧ)
    g¹²_2   = @zeros(nₓ, nᵧ)
    g²¹_2   = @zeros(nₓ, nᵧ)
    g²²_2   = @ones(nₓ, nᵧ)

    # ------------------------------------------------------------------------------

    x .= repeat((0:(2nₓ))*w/(2nₓ), 1, 2nᵧ+1)
    y .= h/w*x'
    
    @. x˒ξ_1[2:end-1,2:end-1] = (x[4:2:end-3,4:2:end-3] - x[2:2:end-5,4:2:end-3])/dξ
    @. y˒ξ_1[2:end-1,2:end-1] = (y[4:2:end-3,4:2:end-3] - y[2:2:end-5,4:2:end-3])/dξ

    @. x˒ξ_2[2:end-1,2:end-1] = (x[5:2:end-2,3:2:end-4] - x[3:2:end-4,3:2:end-4])/dξ
    @. y˒ξ_2[2:end-1,2:end-1] = (y[5:2:end-2,3:2:end-4] - y[3:2:end-4,3:2:end-4])/dξ

    @. x˒η_1[2:end-1,2:end-1] = (x[3:2:end-4,5:2:end-2] - x[3:2:end-4,3:2:end-4])/dη
    @. y˒η_1[2:end-1,2:end-1] = (y[3:2:end-4,5:2:end-2] - y[3:2:end-4,3:2:end-4])/dη    

    @. x˒η_2[2:end-1,2:end-1] = (x[4:2:end-3,5:2:end-2] - x[4:2:end-3,3:2:end-4])/dη
    @. y˒η_2[2:end-1,2:end-1] = (y[4:2:end-3,5:2:end-2] - y[4:2:end-3,3:2:end-4])/dη

    @. inv_J_1 = x˒ξ_1 * y˒η_1 - y˒ξ_1 * x˒η_1
    @. inv_J_2 = x˒ξ_2 * y˒η_2 - y˒ξ_2 * x˒η_2    

    @. J_1 = 1 / inv_J_1
    @. J_2 = 1 / inv_J_2

    @. g¹ₓ_1 =  J_1*y˒η_1
    @. g¹ᵧ_1 = -J_1*x˒η_1
    @. g²ₓ_1 = -J_1*y˒ξ_1
    @. g²ᵧ_1 = -J_1*x˒ξ_1

    @. g¹ₓ_2 =  J_2*y˒η_2
    @. g¹ᵧ_2 = -J_2*x˒η_2
    @. g²ₓ_2 = -J_2*y˒ξ_2
    @. g²ᵧ_2 = -J_2*x˒ξ_2

    g₁ₓ_1 = x˒ξ_1
    g₁ᵧ_1 = y˒ξ_1
    g₁ₓ_2 = x˒ξ_2
    g₁ᵧ_2 = y˒ξ_2

    g₂ₓ_1 = x˒η_1
    g₂ᵧ_1 = y˒η_1    
    g₂ₓ_2 = x˒η_2
    g₂ᵧ_2 = y˒η_2

    @. g¹¹_1 = g¹ₓ_1^2 + g¹ᵧ_1^2
    @. g¹²_1 = g¹ₓ_1*g²ₓ_1 + g¹ᵧ_1*g²ᵧ_1
    @. g²¹_1 = g¹²_1
    @. g²²_1 = g²ₓ_1^2 + g²ᵧ_1^2

    @. g¹¹_2 = g¹ₓ_2^2 + g¹ᵧ_2^2
    @. g¹²_2 = g¹ₓ_2*g²ₓ_2 + g¹ᵧ_2*g²ᵧ_2
    @. g²¹_2 = g¹²_2
    @. g²²_2 = g²ₓ_2^2 + g²ᵧ_2^2

    Δ = zeros(nₓ, nᵧ, nₓ, nᵧ)
    for i∈1:nₓ, j∈1:nᵧ
        J = J_1[i,j]
        #J = 1/4*(J_1[i,j]+J_1[i+1,j]+J_2[i,j]+J_2[i,j+1])

        if i<nₓ
            Δ[i,j, i+1, j] += J/dξ^2*g¹¹_1[i+1,j]*inv_J_1[i+1,j]
            Δ[i,j,   i, j] -= J/dξ^2*g¹¹_1[i+1,j]*inv_J_1[i+1,j]
        end
        if  1<i
            Δ[i,j,   i, j] -= J/dξ^2*g¹¹_1[  i,j]*inv_J_1[  i,j]
            Δ[i,j, i-1, j] += J/dξ^2*g¹¹_1[  i,j]*inv_J_1[  i,j]
        end


        if j<nᵧ
            Δ[i,j, i, j+1] += J/dη^2*g²²_2[i,j+1]*inv_J_2[i,j+1]
            Δ[i,j, i,   j] -= J/dη^2*g²²_2[i,j+1]*inv_J_2[i,j+1]            
        end
        if 1<j
            Δ[i,j, i,   j] -= J/dη^2*g²²_2[i,  j]*inv_J_2[i,  j]
            Δ[i,j, i, j-1] += J/dη^2*g²²_2[i,  j]*inv_J_2[i,  j]            
        end

        if i<nₓ && 1<j<nᵧ
            #Δ[i,j, i  ,  j] += (1/4)*J/(dξ*dη)*g¹²_1[i+1,j]
            #Δ[i,j, i+1,  j] += (1/4)*J/(dξ*dη)*g¹²_1[i+1,j]
            Δ[i,j, i  ,j+1] += (1/4)*J/(dξ*dη)*g¹²_1[i+1,j]
            Δ[i,j, i+1,j+1] += (1/4)*J/(dξ*dη)*g¹²_1[i+1,j]
            
            #Δ[i,j, i  ,  j] -= (1/4)*J/(dξ*dη)*g¹²_1[i+1,j]
            #Δ[i,j, i+1,  j] -= (1/4)*J/(dξ*dη)*g¹²_1[i+1,j]
            Δ[i,j, i  ,j-1] -= (1/4)*J/(dξ*dη)*g¹²_1[i+1,j]
            Δ[i,j, i+1,j-1] -= (1/4)*J/(dξ*dη)*g¹²_1[i+1,j]            
            
        end

        if 1<i && 1<j<nᵧ
            #Δ[i,j, i  ,j  ] -= (1/4)*J/(dξ*dη)*g¹²_1[i,j]
            #Δ[i,j, i-1,j  ] -= (1/4)*J/(dξ*dη)*g¹²_1[i,j]
            Δ[i,j, i  ,j+1] -= (1/4)*J/(dξ*dη)*g¹²_1[i,j]
            Δ[i,j, i-1,j+1] -= (1/4)*J/(dξ*dη)*g¹²_1[i,j]

            #Δ[i,j, i  ,j  ] += (1/4)*J/(dξ*dη)*g¹²_1[i,j]
            #Δ[i,j, i-1,j  ] += (1/4)*J/(dξ*dη)*g¹²_1[i,j]
            Δ[i,j, i  ,j+1] += (1/4)*J/(dξ*dη)*g¹²_1[i,j]
            Δ[i,j, i-1,j+1] += (1/4)*J/(dξ*dη)*g¹²_1[i,j]                        
        end

        if 1<i<nₓ && j<nᵧ
            #Δ[i,j,  i,  j] += (1/4)*J/(dη*dξ)*g²¹_2[i,j+1]
            Δ[i,j,i+1,  j] += (1/4)*J/(dη*dξ)*g²¹_2[i,j+1]
           # Δ[i,j,  i,j+1] += (1/4)*J/(dη*dξ)*g²¹_2[i,j+1]
            Δ[i,j,i+1,j+1] += (1/4)*J/(dη*dξ)*g²¹_2[i,j+1]

           # Δ[i,j,  i,  j] -= (1/4)*J/(dη*dξ)*g²¹_2[i,j+1]
            Δ[i,j,i-1,  j] -= (1/4)*J/(dη*dξ)*g²¹_2[i,j+1]
            #Δ[i,j,  i,j+1] -= (1/4)*J/(dη*dξ)*g²¹_2[i,j+1]
            Δ[i,j,i-1,j+1] -= (1/4)*J/(dη*dξ)*g²¹_2[i,j+1]                        
        end

        if 1<i<nₓ && 1<j
            #Δ[i,j,  i,  j] -= (1/4)*J/(dη*dξ)*g²¹_2[i,j]
            Δ[i,j,i+1,  j] -= (1/4)*J/(dη*dξ)*g²¹_2[i,j]
            #Δ[i,j,  i,j-1] -= (1/4)*J/(dη*dξ)*g²¹_2[i,j]
            Δ[i,j,i+1,j-1] -= (1/4)*J/(dη*dξ)*g²¹_2[i,j]

            #Δ[i,j,  i,  j] += (1/4)*J/(dη*dξ)*g²¹_2[i,j]
            Δ[i,j,i-1,  j] += (1/4)*J/(dη*dξ)*g²¹_2[i,j]
            #Δ[i,j,  i,j-1] += (1/4)*J/(dη*dξ)*g²¹_2[i,j]
            Δ[i,j,i-1,j-1] += (1/4)*J/(dη*dξ)*g²¹_2[i,j]            
        end    
    end
    
    Δ = reshape(Δ, nₓ*nᵧ, nₓ*nᵧ)
    # ------------------------------------------------------------------------------

    for i∈1:nₜ
        compute_u★!(nₓ, nᵧ,
                    dt, dξ, dη,
                    ν,
                    J_1, J_2, inv_J_1, inv_J_2,
                    g¹¹_1, g¹²_1, g²¹_1, g²²_1, g¹¹_2, g¹²_2, g²¹_2, g²²_2,
                    g¹ₓ_1, g¹ᵧ_1, g²ₓ_2, g²ᵧ_2,                    
                    uₓ_1, uᵧ_1, uₓ_2, uᵧ_2, u¹_1, u²_1, u¹_2, u²_2, u★¹_1, u★²_2)
        
        compute_div_u★!(nₓ, nᵧ,
                       dt, dξ, dη,
                       ρ,
                       J_1, J_2, inv_J_1, inv_J_2,
                       u★¹_1, u★²_2, div_u★)

        p = reshape(Δ \ reshape(div_u★, nₓ*nᵧ), nₓ, nᵧ)

        update_u!(nₓ, nᵧ,
                  dt, dξ, dη,
                  ρ,
                  g¹¹_1, g¹²_1, g²¹_1, g²²_1, g¹¹_2, g¹²_2, g²¹_2, g²²_2, g₁ₓ_1, g₁ᵧ_1, g₁ₓ_2, g₁ᵧ_2, g₂ₓ_1, g₂ᵧ_1, g₂ₓ_2, g₂ᵧ_2,
                   u★¹_1, u★²_2,
                   p,
                   u¹_1, u²_1, u¹_2, u²_2, uₓ_1, uᵧ_1, uₓ_2, uᵧ_2)        
        
    end
        
    return nₓ, nᵧ, uₓ_1, uᵧ_2
end

nₓ, nᵧ, uₓ_1, uᵧ_2 = NS_solve()

x = repeat((0:(nₓ-1))*1/nₓ, 1, nᵧ)
y = x'

# egv = eigvecs(Δ)

#PyPlot.plot_surface(x, y, reshape(egv[:,end], nₓ, nᵧ))
