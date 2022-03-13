using LinearAlgebra
using SparseArrays
using Plots
const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);
end 

function ns_step!(u_x, u_y, u_z, u_x2, u_y2, u_z2, u_xstar, u_ystar, u_zstar, div_u_star, p, p2, ν, rho, dt, dx, dy, dz, nx, ny, nz, Δ)

    @parallel update_ustar!(u_x, u_y, u_z, u_xstar, u_ystar, u_zstar, ν, dx, dy, dz, dt)
    # boundary conditions on u_star ?
    
    @parallel compute_div_u_star!(u_xstar, u_ystar, u_zstar, div_u_star, dx, dy, dz)
    # boundary conditions on div_u_star ?
    @parallel dirichlet_bc_x!(div_u_star, 0, 0)
    @parallel dirichlet_bc_y!(div_u_star, 0, 0)
    @parallel dirichlet_bc_z!(div_u_star, 0, 0)    

    # initialize p
    # p .= 0.0
    # ε = 1e-2
    # dτ=0.00001
    # err = Inf
    # nb_iter = 0
    # while err > ε
    #     @parallel update_pressure!(div_u_star, dτ, dt, dx, dy, dz, rho, p, p2)

    #     @parallel neumann_bc_x!(p2)
    #     @parallel neumann_bc_y!(p2)
    #     @parallel neumann_bc_z!(p2)
        
    #     p, p2 = p2, p
    #     err = norm(p-p2)
    #     nb_iter += 1
    # end
    # println(err, "  ", nb_iter)
    p = reshape(Δ \ reshape(-rho/dt*div_u_star, nx*ny*nz), nx, ny, nz)
    
    @parallel update_u!(u_xstar, u_ystar, u_zstar, u_x2, u_y2, u_z2, p, rho, dt, dx, dy, dz)
    
    @parallel dirichlet_bc_x!(u_x2, 0, 0)
    @parallel dirichlet_bc_x!(u_y2, 0, 0)
    @parallel dirichlet_bc_x!(u_z2, 0, 0)

    @parallel dirichlet_bc_y!(u_x2, 0, 0)
    @parallel dirichlet_bc_y!(u_y2, 0, 0)
    @parallel dirichlet_bc_y!(u_z2, 0, 0)

    u = 1
    @parallel dirichlet_bc_z!(u_x2, 0, u)
    @parallel dirichlet_bc_z!(u_y2, 0, 0)
    @parallel dirichlet_bc_z!(u_z2, 0, 0)    
end

@parallel_indices (ix, iy, iz) function neumann_bc_x!(A)
    A[1,iy,iz] = A[2,iy,iz]
    A[end,iy,iz] = A[end-1,iy,iz]
    return nothing
end

@parallel_indices (ix, iy, iz) function neumann_bc_y!(A)
    A[ix,1,iz] = A[ix,2,iz]
    A[ix,end,iz] = A[ix,end-1,iz]
    return nothing
end

@parallel_indices (ix, iy, iz) function neumann_bc_z!(A)
    A[ix,iy,1] = A[ix,iy,2]
    A[ix,iy,end] = A[ix,iy,end-1]
    return nothing
end

@parallel_indices (ix, iy, iz) function dirichlet_bc_x!(A, l, h)
    A[1,iy,iz] = l
    A[end,iy,iz] = h
    return nothing
end

@parallel_indices (ix, iy, iz) function dirichlet_bc_y!(A, l, h)
    A[ix,1,iz] = l
    A[ix,end,iz] = h
    return nothing
end

@parallel_indices (ix, iy, iz) function dirichlet_bc_z!(A, l, h)
    A[ix,iy,1] = l
    A[ix,iy,end] = h
    return nothing
end
          

@parallel_indices (ix,iy,iz) function update_ustar!(u_x, u_y, u_z, u_xstar, u_ystar, u_zstar, ν, dx, dy, dz, dt)
    if ix>1 && ix<size(u_x,1) && iy>1 && iy<size(u_x,2) && iz>1 && iz<size(u_x,3)
        u_xstar[ix,iy,iz] = u_x[ix,iy,iz] + dt*(ν*((u_x[ix+1,iy,iz]+u_x[ix-1,iy,iz]-2u_x[ix,iy,iz])/dx^2+(u_x[ix,iy+1,iz]+u_x[ix,iy-1,iz]-2u_x[ix,iy,iz])/dy^2+(u_x[ix,iy,iz+1]+u_x[ix,iy,iz-1]-2u_x[ix,iy,iz])/dz^2)-
            (u_x[ix,iy,iz]*(u_x[ix+1,iy,iz]-u_x[ix-1,iy,iz])/(2dx)+(u_y[ix-1,iy+1,iz]+u_y[ix-1,iy,iz]+u_y[ix,iy+1,iz]+u_y[ix,iy,iz])/4*(u_x[ix,iy+1,iz]-u_x[ix,iy-1,iz])/(2dy)+(u_z[ix-1,iy+1,iz+1]+u_z[ix-1,iy+1,iz]+u_z[ix-1,iy,iz+1]+u_z[ix-1,iy,iz])/4*(u_x[ix,iy,iz+1]-u_x[ix,iy,iz+1])/(2dz)))
        u_ystar[ix,iy,iz] = u_y[ix,iy,iz] + dt*(ν*((u_y[ix+1,iy,iz]+u_y[ix-1,iy,iz]-2u_y[ix,iy,iz])/dx^2+(u_y[ix,iy+1,iz]+u_y[ix,iy-1,iz]-2u_y[ix,iy,iz])/dy^2+(u_y[ix,iy,iz+1]+u_y[ix,iy,iz-1]-2u_y[ix,iy,iz])/dz^2)-
            ((u_x[ix+1,iy,iz]+u_x[ix+1,iy-1,iz]+u_x[ix,iy,iz]+u_x[ix-1,iy,iz])/4*(u_y[ix+1,iy,iz]-u_y[ix-1,iy,iz])/(2dx)+u_y[ix,iy,iz]*(u_y[ix,iy+1,iz]-u_y[ix,iy-1,iz])/(2dy)+(u_z[ix,iy,iz+1]+u_z[ix,iy-1,iz+1]+u_z[ix,iy,iz]+u_z[ix,iy-1,iz])/4*(u_y[ix,iy,iz+1]-u_y[ix,iy,iz-1])/(2dz)))
        u_zstar[ix,iy,iz] = u_z[ix,iy,iz] + dt*(ν*((u_z[ix+1,iy,iz]+u_z[ix-1,iy,iz]-2u_z[ix,iy,iz])/dx^2+(u_z[ix,iy+1,iz]+u_z[ix,iy-1,iz]-2u_z[ix,iy,iz])/dy^2+(u_z[ix,iy,iz+1]+u_z[ix,iy,iz-1]-2u_z[ix,iy,iz])/dz^2)-
            ((u_x[ix+1,iy,iz]+u_x[ix+1,iy,iz-1]+u_x[ix,iy,iz]+u_x[ix,iy,iz-1])/4*(u_z[ix+1,iy,iz]-u_z[ix-1,iy,iz])/(2dx)+(u_y[ix,iy+1,iz]+u_y[ix,iy+1,iz-1]+u_y[ix,iy,iz]+u_y[ix,iy,iz-1])/4*(u_z[ix,iy+1,iz]-u_z[ix,iy-1,iz])/(2dy)+u_z[ix,iy,iz]*(u_z[ix,iy,iz+1]-u_z[ix,iy,iz-1])/(2dz)))
    end
    return nothing
end

@parallel_indices (ix, iy, iz) function compute_div_u_star!(u_xstar, u_ystar, u_zstar, div_u_star, dx, dy, dz)
    if ix>1 && ix<size(u_xstar,1) && iy>1 && iy<size(u_xstar,2) && iz>1 && iz<size(u_xstar,3)
        div_u_star[ix,iy,iz] = (u_xstar[ix+1,iy,iz]-u_xstar[ix,iy,iz])/dx + (u_ystar[ix,iy+1,iz]-u_ystar[ix,iy,iz])/dy + (u_zstar[ix,iy,iz+1]-u_zstar[ix,iy,iz])/dz
    end
    return nothing
end

@parallel_indices (ix, iy, iz) function update_u!(u_xstar, u_ystar, u_zstar, u_x2, u_y2, u_z2, p, rho, dt, dx, dy, dz)
    if ix>1 && ix<size(u_xstar,1) && iy>1 && iy<size(u_xstar,2) && iz>1 && iz<size(u_xstar,3)
        u_x2[ix,iy,iz] = u_xstar[ix,iy,iz] - dt/rho*(p[ix+1,iy,iz]-p[ix,iy,iz])/dx
        u_y2[ix,iy,iz] = u_ystar[ix,iy,iz] - dt/rho*(p[ix,iy+1,iz]-p[ix,iy,iz])/dy
        u_z2[ix,iy,iz] = u_zstar[ix,iy,iz] - dt/rho*(p[ix,iy,iz+1]-p[ix,iy,iz])/dz                
    end
    return nothing
end

@parallel_indices (ix, iy, iz) function update_pressure!(div_u_star, dτ, dt, dx, dy, dz, rho, p, p2)
    if ix>1 && ix<size(p,1) && iy>1 && iy<size(p,2) && iz>1 && iz<size(p,3)
        p2[ix,iy,iz] = p[ix,iy,iz] - dτ*((p[ix+1,iy,iz]+p[ix-1,iy,iz]-2p[ix,iy,iz])/dx^2+(p[ix,iy+1,iz]+p[ix,iy-1,iz]-2p[ix,iy,iz])/dy^2+(p[ix,iy,iz+1]+p[ix,iy,iz-1]-2p[ix,iy,iz])/dz^2 + rho/dt*div_u_star[ix,iy,iz])        
    end
    return nothing
end

function ns3D()
    # Physics
    rho = 1.0
    ν = 1.0
    lx, ly, lz = 1.0, 1.0, 1.0

    # Numerics
    nx, ny, nz = 32, 32, 32
    nt = 100
    dx = lx/(nx-1)
    dy = ly/(ny-1)
    dz = lz/(nz-1)

    # Array initializations
    u_x = @zeros(nx, ny, nz)
    u_y = @zeros(nx, ny, nz)
    u_z = @zeros(nx, ny, nz)
    u_x2 = @zeros(nx, ny, nz)
    u_y2 = @zeros(nx, ny, nz)
    u_z2 = @zeros(nx, ny, nz)
    u_xstar = @zeros(nx, ny, nz)
    u_ystar = @zeros(nx, ny, nz)
    u_zstar = @zeros(nx, ny, nz)    
    div_u_star = @zeros(nx, ny, nz) 
    p = @zeros(nx, ny, nz)
    p2 = @zeros(nx, ny, nz)

    # Time loop
    dt = 1e-4#min(dx^2,dy^2,dz^2);

    ∂_x2 = spdiagm(0 => -2ones(nx), 1 => ones(nx-1), -1 => ones(nx-1))/dx^2
    # ∂_x2[1,2] = 2/dx^2
    # ∂_x2[end,end-1] = 2/dx^2
    
    ∂_y2 = spdiagm(0 => -2ones(ny), 1 => ones(ny-1), -1 => ones(ny-1))/dy^2
    # ∂_y2[1,2] = 2/dy^2
    # ∂_y2[end,end-1] = 2/dy^2
    
    ∂_z2 = spdiagm(0 => -2ones(nz), 1 => ones(nz-1), -1 => ones(nz-1))/dz^2
    # ∂_z2[1,2] = 2/dz^2
    # ∂_z2[end,end-1] = 2/dz^2

    Δ = kron(∂_x2, I(ny), I(nz)) + kron(I(nx), ∂_y2, I(nz)) + kron(I(nx), I(ny), ∂_z2)
    # Δ[1,:,:] = ones(nx*ny*nz)
    # Δ[1,1,1] = 1/dx^2
    
    for it = 1:nt
        ns_step!(u_x, u_y, u_z, u_x2, u_y2, u_z2, u_xstar, u_ystar, u_zstar, div_u_star, p, p2, ν, rho, dt, dx, dy, dz, nx, ny, nz, Δ)
        u_x, u_x2 = u_x2, u_x
        u_y, u_y2 = u_y2, u_y
        u_z, u_z2 = u_z2, u_z
        println(it)
    end

    return u_x, u_y, u_z, p
end

u_x, u_y, u_z, p = ns3D()
u = u_x .^2 .+ u_y .^2 .+ u_z .^2
