using LinearAlgebra
using SparseArrays
using PyPlot

const USE_GPU = false
using ParallelStencil
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);
end

@parallel_indices (i,j,k) function update_u_star!(u_x_star, u_y_star, u_z_star, u_x, u_y, u_z, ν, dx, dy, dz, dt, nx, ny, nz)
    if i>1 && i<=nx-1 && j>1 && j<= ny-1 && k>1 && k<nz-1
        u_x_star[i,j,k] = u_x[i,j,k] + dt*(-(u_x[i,j,k]*(u_x[i+1,j,k]-u_x[i-1,j,k])/(2dx)+(u_y[i,j,k]+u_y[i,j+1,k]+u_y[i-1,j+1,k]+u_y[i-1,j,k])/4*(u_x[i,j+1,k]-u_x[i,j-1,k])/(2dy)+(u_z[i-1,j,k+1]+u_z[i,j,k+1]+u_z[i-1,j,k]+u_z[i,j,k])/4*(u_x[i,j,k+1]-u_x[i,j,k-1])/(2dz))+ν*((u_x[i+1,j,k]+u_x[i-1,j,k]-2*u_x[i,j,k])/(dx^2)+(u_x[i,j+1,k]+u_x[i,j-1,k]-2*u_x[i,j,k])/(dy^2)+(u_x[i,j,k+1]+u_x[i,j,k-1]-2*u_x[i,j,k])/(dz^2)))
        u_y_star[i,j,k] = u_y[i,j,k] + dt*(-((u_x[i,j,k]+u_x[i+1,j,k]+u_x[i+1,j-1,k]+u_x[i,j-1,k])/4*(u_y[i+1,j,k]-u_y[i-1,j,k])/(2dx)+u_y[i,j,k]*(u_y[i,j+1,k]-u_y[i,j-1,k])/(2dy)+(u_z[i,j-1,k+1]+u_z[i,j,k+1]+u_z[i,j-1,k]+u_z[i,j,k])/4*(u_y[i,j,k+1]-u_y[i,j,k-1])/(2dz))+ν*((u_y[i+1,j,k]+u_y[i-1,j,k]-2*u_y[i,j,k])/(dx^2)+(u_y[i,j+1,k]+u_y[i,j-1,k]-2*u_y[i,j,k])/(dy^2)+(u_y[i,j,k+1]+u_y[i,j,k-1]-2*u_y[i,j,k])/(dz^2)))
        u_z_star[i,j,k] = u_z[i,j,k] + dt*(-((u_x[i+1,j,k-1]+u_x[i,j,k-1]+u_x[i+1,j,k]+u_x[i,j,k])/4*(u_z[i+1,j,k]-u_z[i-1,j,k])/(2dx)+(u_y[i,j+1,k-1]+u_y[i,j+1,k]+u_y[i,j,k-1]+u_y[i,j,k])/4*(u_z[i,j+1,k]-u_z[i,j-1,k])/(2dy)+u_z[i,j,k]*(u_z[i,j,k+1]-u_z[i,j,k-1])/(2dz))+ν*((u_z[i+1,j,k]+u_z[i-1,j,k]-2*u_z[i,j,k])/(dx^2)+(u_z[i,j+1,k]+u_z[i,j-1,k]-2*u_z[i,j,k])/(dy^2)+(u_z[i,j,k+1]+u_z[i,j,k-1]-2*u_z[i,j,k])/(dz^2)))
    end
    return nothing
end

@parallel_indices (i,j,k) function update_div_u_star!(div_u_star, u_x_star, u_y_star, u_z_star, dx, dy, dz, nx, ny, nz)
    if i>1 && i<=nx-1 && j>1 && j<= ny-1 && k>1 && k<nz-1
        div_u_star[i,j,k] = (u_x_star[i+1,j,k]-u_x_star[i,j,k])/dx + (u_y_star[i,j+1,k]-u_y_star[i,j,k])/dy + (u_z_star[i,j,k+1]-u_z_star[i,j,k])/dz
    end
    return nothing
end

function update_p!(p, div_u_star, Δ, ρ, dx, dy, dz, dt, nx, ny, nz)
    p .= reshape(Δ \ reshape((ρ/dt)*div_u_star, nx*ny*nz), nx, ny, nz)
end

@parallel_indices (i,j,k) function update_u!(u_x, u_y, u_z, u_x_star, u_y_star, u_z_star, p, ρ, dx, dy, dz, dt, nx, ny, nz)
    if i>1 && i<=nx-1 && j>1 && j<= ny-1 && k>1 && k<nz-1
        u_x[i,j,k] = u_x_star[i,j,k] -(dt/ρ)*(p[i,j,k]-p[i-1,j,k])/dx
        u_y[i,j,k] = u_y_star[i,j,k] -(dt/ρ)*(p[i,j,k]-p[i,j-1,k])/dy
        u_z[i,j,k] = u_z_star[i,j,k] -(dt/ρ)*(p[i,j,k]-p[i,j,k-1])/dz
    end
    return nothing
end

function NS_solve()
    nx, ny, nz, nt = 30, 30, 30, 1000

    w, h, d = 1.0, 1.0, 1.0
    dx, dy, dz = w/nx, h/ny, d/nz
    dt = min(dx^2, dy^2, dz^2)
    ρ = 1.0
    ν = 0.01

    u_x        = @zeros(nx, ny, nz)
    u_y        = @zeros(nx, ny, nz)
    u_z        = @zeros(nx, ny, nz)    
    u_x_star   = @zeros(nx, ny, nz)
    u_y_star   = @zeros(nx, ny, nz)
    u_z_star   = @zeros(nx, ny, nz)    
    div_u_star = @zeros(nx, ny, nz)
    p          = @zeros(nx, ny, nz)

    U = 1.0
    Us = U*ones(nx, nz)
    Us[1] = 0.0
    Us[end] = 0.0

    ∂_x2 = spdiagm(0 => -2/(dx^2)*ones(nx),
                   1 =>  1/(dx^2)*ones(nx-1),
                  -1 =>  1/(dx^2)*ones(nx-1))
    ∂_x2[1,2] = 2/(dx^2) # ∂p/∂x = 0 at y = 0
    # p = 0 at y=h

    ∂_y2 = spdiagm(0 => -2/(dy^2)*ones(ny),
                   1 =>  1/(dy^2)*ones(ny-1),
                  -1 =>  1/(dy^2)*ones(ny-1))
    ∂_y2[1,2] = 2/(dy^2) # ∂p/∂y = 0 at x = 0
    ∂_y2[end,end-1] = 2/(dy^2) # ∂p/∂y = 0 at x=w

    ∂_z2 = spdiagm(0 => -2/(dz^2)*ones(nz),
                   1 =>  1/(dz^2)*ones(nz-1),
                  -1 =>  1/(dz^2)*ones(nz-1))
    ∂_z2[1,end] = 1/(dz^2)
    ∂_z2[end,1] = 1/(dz^2)    

    Δ = kron(∂_x2, I(ny), I(nz)) + kron(I(nx), ∂_y2, I(nz)) + kron(I(nx), I(ny), ∂_z2)

    for n∈1:nt
        println(n)
        
        @parallel update_u_star!(u_x_star, u_y_star, u_z_star, u_x, u_y, u_z, ν, dx, dy, dz, dt, nx, ny, nz)
        u_x_star[1,:,:] .= 0
        u_x_star[end,:,:] .= 0

        u_y_star[1,:,:] .= -u_y_star[2,:,:]
        u_y_star[end,:,:] .= -u_y_star[end-1,:,:]

        u_z_star[1,:,:] = -u_z_star[2,:,:]
        u_z_star[end,:,:] = -u_z_star[end-1,:,:]

        u_x_star[:,1,:] .= -u_x_star[:,2,:]
        u_x_star[:,end,:] .= -u_x_star[:,end-1,:]

        u_y_star[:,1,:] .= 0
        u_y_star[:,end,:] .= 0

        u_z_star[:,1,:] = -u_z_star[:,2,:]
        u_z_star[:,end,:] = -u_z_star[:,end-1,:]

        u_x_star[:,:,1] = u_x_star[:,:,end-1]
        u_x_star[:,:,end] = u_x_star[:,:,2]

        u_y_star[:,:,1] = u_y_star[:,:,end-1]
        u_y_star[:,:,end] = u_y_star[:,:,2]

        u_z_star[:,:,1] = u_z_star[:,:,end-1]
        u_z_star[:,:,end] = u_z_star[:,:,2]
        
        @parallel update_div_u_star!(div_u_star, u_x_star, u_y_star, u_z_star, dx, dy, dz, nx, ny, nz)
        div_u_star[1,:,:] .= -div_u_star[2,:,:]
        div_u_star[end,:,:] .= -div_u_star[end-1,:,:]

        div_u_star[:,1,:] .= -div_u_star[:,2,:]
        div_u_star[:,end,:] .= -div_u_star[:,end-1,:]

        div_u_star[:,:,1] .= div_u_star[:,:,end-1]
        div_u_star[:,:,end] .= div_u_star[:,:,2]
        
        update_p!(p, div_u_star, Δ, ρ, dx, dy, dz, dt, nx, ny, nz)
        @parallel update_u!(u_x, u_y, u_z, u_x_star, u_y_star, u_z_star, p, ρ, dx, dy, dz, dt, nx, ny, nz)

        u_x[1,:,:] .= 0
        u_x[end,:,:] .= 0

        u_y[1,:,:] .= -u_y[2,:,:]
        u_y[end,:,:] .= -u_y[end-1,:,:]

        u_z[1,:,:] .= -u_z[2,:,:]
        u_z[end,:,:] .= -u_z[end-1,:,:]

        u_x[:,1,:] .= -u_x[:,2,:]
        u_x[:,end,:] .= 2*Us-u_x[:,end-1,:]

        u_y[:,1,:] .= 0
        u_y[:,end,:] .= 0

        u_z[:,1,:] .= -u_z[:,2,:]
        u_z[:,end,:] .= u_z[:,end-1,:]

        u_x[:,:,1] = u_x[:,:,end-1]
        u_x[:,:,end] = u_x[:,:,2]

        u_y[:,:,1] = u_y[:,:,end-1]
        u_y[:,:,end] = u_y[:,:,2]

        u_z[:,:,1] = u_z[:,:,end-1]
        u_z[:,:,end] = u_z[:,:,2]
    end

    u = sqrt.(u_x .^2 .+ u_y .^2 .+ u_z .^2)
    
    return u_x, u_y, u_z, u, p, nx, ny, nz, dx, dy, dz, w, h, d
end

u_x, u_y, u_z, u, p, nx, ny, nz, dx, dy, dz, w, h = NS_solve()

x_v_domain = (0:(nx-2))*w/(nx-2)
y_v_domain = (0:(ny-2))*h/(ny-2) 
x_v = repeat(x_v_domain', ny-1)
y_v = repeat(y_v_domain, 1, nx-1)

u_x_avg = (u_x[2:end,1:end-1,:] .+ u_x[2:end,2:end,:]) ./ 2
u_y_avg = (u_y[1:end-1,2:end,:] .+ u_y[2:end,2:end,:]) ./ 2

x_p = (0:(nx-1))*w/(nx-1)
y_p = (0:(ny-1))*h/(ny-1)

PyPlot.contourf(x_p, y_p, p[:,:,div(nz,2)]')
PyPlot.colorbar()
PyPlot.streamplot(x_v, y_v, u_x_avg[:,:,div(nz,2)]', u_y_avg[:,:,div(nz,2)]', color="black")

