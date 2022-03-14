using LinearAlgebra
using SparseArrays
using Plots
using LinearAlgebra
using SparseArrays
using Plots
const USE_GPU = false
using ParallelStencil
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2);
else
    @init_parallel_stencil(Threads, Float64, 2);
end

@parallel_indices (i,j) function update_u_star!(u_x_star, u_y_star, u_x, u_y, ν, dx, dy, dt, nx, ny)
    if i>1 && i<=nx-1 && j>1 && j<= ny-1
        u_x_star[i,j] = u_x[i,j] + dt*(-(u_x[i,j]*(u_x[i+1,j]-u_x[i-1,j])/(2dx)+(u_y[i,j]+u_y[i,j+1]+u_y[i-1,j+1]+u_y[i-1,j])/4*(u_x[i,j+1]-u_x[i,j-1])/(2dy))+ν*((u_x[i+1,j]+u_x[i-1,j]-2*u_x[i,j])/(dx^2)+(u_x[i,j+1]+u_y[i,j-1]-2*u_y[i,j])/(dy^2)))
        u_y_star[i,j] = u_y[i,j] + dt*(-((u_x[i,j]+u_x[i+1,j]+u_x[i+1,j-1]+u_x[i,j-1])/4*(u_y[i+1,j]-u_y[i-1,j])/(2dx)+u_y[i,j]*(u_y[i,j+1]-u_y[i,j-1])/(2dy))+ν*((u_y[i+1,j]+u_y[i-1,j]-2*u_y[i,j])/(dx^2)+(u_y[i,j+1]+u_y[i,j-1]-2*u_y[i,j])/(dy^2)))
    end
    return nothing
end

@parallel_indices (i,j) function update_div_u_star!(div_u_star, u_x_star, u_y_star, dx, dy, nx, ny)
    if i>1 && i<=nx-1 && j>1 && j<= ny-1        
        div_u_star[i,j] = (u_x_star[i+1,j]-u_x_star[i,j])/dx + (u_y_star[i,j+1]-u_y_star[i,j])/dy
    end
    return nothing
end

function update_p!(p, div_u_star, Δ, ρ, dx, dy, dt, nx, ny)
    p .= reshape(Δ \ reshape((ρ/dt)*div_u_star, nx*ny), nx, ny)
end

@parallel_indices (i,j) function update_u!(u_x, u_y, u_x_star, u_y_star, p, ρ, dx, dy, dt, nx, ny, F)
    if i>1 && i<=nx-1 && j>1 && j<= ny-1    
        u_x[i,j] = u_x_star[i,j] -(dt/ρ)*(p[i,j]-p[i-1,j])/dx + dt*F
        u_y[i,j] = u_y_star[i,j] -(dt/ρ)*(p[i,j]-p[i,j-1])/dy
    end
    return nothing
end

function NS_solve()
    nx, ny, nt = 64, 64, 1000

    w, h = 1.0, 1.0
    dx, dy = w/nx, h/ny
    dt = 0.00001
    ρ = 1.0
    ν = 0.1

    u_x        = zeros(nx, ny)
    u_y        = zeros(nx, ny)
    u_x_star   = zeros(nx, ny)
    u_y_star   = zeros(nx, ny)
    div_u_star = zeros(nx, ny)
    p          = zeros(nx, ny)

    # U = 1.0
    F = 1.0

    ∂_x2 = spdiagm(0 => -2/(dx^2)*ones(nx),
                   1 =>  1/(dx^2)*ones(nx-1),
                   -1 =>  1/(dx^2)*ones(nx-1))
    ∂_x2[1,end] = 1/(dx^2)
    ∂_x2[end,1] = 1/(dx^2)

    ∂_y2 = spdiagm(0 => -2/(dy^2)*ones(ny),
                   1 =>  1/(dy^2)*ones(ny-1),
                   -1 =>  1/(dy^2)*ones(ny-1))
    ∂_y2[1,2] = 2/(dy^2)
    ∂_y2[end,end-1] = 2/(dy^2)

    Δ = kron(∂_x2, I(ny)) + kron(I(nx), ∂_y2)
    # Δ[1,:] .= 0
    # Δ[1,1] = 1
    
    for n∈1:nt
        println(n)
        u_x[1,:] = u_x[end-1,:]
        u_y[1,:] = u_y[end-1,:]

        u_x[end,:] = u_x[2,:]
        u_y[end,:] = u_y[2,:]

        u_x[:,1] = -u_x[:,2]
        u_x[:,end] = -u_x[:,end-1]
        
        @parallel update_u_star!(u_x_star, u_y_star, u_x, u_y, ν, dx, dy, dt, nx, ny)
        u_x_star[1,:] = u_x_star[end-1,:]
        u_y_star[1,:] = u_y_star[end-1,:]

        u_x_star[end,:] = u_x_star[2,:]
        u_y_star[end,:] = u_y_star[2,:]

        u_x_star[:,1] = -u_x_star[:,2]
        u_x_star[:,end] = -u_x_star[:,end-1]
        
        @parallel update_div_u_star!(div_u_star, u_x_star, u_y_star, dx, dy, nx, ny)
        div_u_star[1,:] = div_u_star[end-1,:]
        div_u_star[end,:] = div_u_star[2,:]
        
        update_p!(p, div_u_star, Δ, ρ, dx, dy, dt, nx, ny)
        @parallel update_u!(u_x, u_y, u_x_star, u_y_star, p, ρ, dx, dy, dt, nx, ny, F)
    end

    u = sqrt.(u_x .^2 .+ u_y .^2)
    
    return u_x, u_y, u, p, nx, ny, dx, dy, w, h
end

u_x, u_y, u, p, nx, ny, dx, dy, w, h = NS_solve()
u_max = maximum(u)

xs = repeat((0:nx-1)*w/(nx-1), ny)
ys = reshape(repeat((0:ny-1)*h/(ny-1), 1, nx)', nx*ny)

quiver_plot = contour((0:nx-1)*w/(nx-1), (0:ny-1)*h/(ny-1), reshape(p, nx, ny)', fill=true)
quiver!(xs, ys, quiver=(dx*vec(u_x)/u_max, dy*vec(u_y)/u_max), arrowsize=0.4)

display(quiver_plot)
