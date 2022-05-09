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

@parallel_indices (i,j) function update_u_star!(u_x_star, u_y_star, u_x, u_y, ν, dx, dy, dt, nx, ny)
    if i>1 && i<=nx-1 && j>1 && j<= ny-1
        u_x_star[i,j] = u_x[i,j] + dt*(-(u_x[i,j]*(u_x[i+1,j]-u_x[i-1,j])/(2dx)+(u_y[i,j]+u_y[i,j+1]+u_y[i-1,j+1]+u_y[i-1,j])/4*(u_x[i,j+1]-u_x[i,j-1])/(2dy))+ν*((u_x[i+1,j]+u_x[i-1,j]-2*u_x[i,j])/(dx^2)+(u_x[i,j+1]+u_x[i,j-1]-2*u_x[i,j])/(dy^2)))
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

@parallel_indices (i,j) function update_p!(p, p2, div_u_star, ρ, dx, dy, dt, dτ, nx, ny)
    if i>1 && i<=nx-1 && j>1 && j<= ny-1    
        p2[i,j] += -dτ*(-((p[i+1,j]+p[i-1,j]-2p[i,j])/(dx^2) + (p[i,j+1]+p[i,j-1]-2p[i,j])/(dy^2))   + (ρ/dt)*div_u_star[i,j])
    end
    return nothing
end

@parallel_indices (i,j) function update_u!(u_x, u_y, u_x_star, u_y_star, p, ρ, dx, dy, dt, nx, ny)
    if i>1 && i<=nx-1 && j>1 && j<= ny-1    
        u_x[i,j] = u_x_star[i,j] -(dt/ρ)*(p[i,j]-p[i-1,j])/dx
        u_y[i,j] = u_y_star[i,j] -(dt/ρ)*(p[i,j]-p[i,j-1])/dy
    end
    return nothing
end

function NS_solve()
    nx, ny, nt = 60, 60, 10000

    w, h = 1.0, 1.0
    dx, dy = w/nx, h/ny
    dt = min(dx^2, dy^2)
    ρ = 1.0
    ν = 0.01

    u_x        = @zeros(nx, ny)
    u_y        = @zeros(nx, ny)
    u_x_star   = @zeros(nx, ny)
    u_y_star   = @zeros(nx, ny)
    div_u_star = @zeros(nx, ny)
    p          = @zeros(nx, ny)
    p2         = @zeros(nx, ny)    

    U = 1.0
    Us = U*ones(nx)
    Us[1] = 0.0
    Us[end] = 0.0

    for n∈1:nt
        println(n)
        
        @parallel update_u_star!(u_x_star, u_y_star, u_x, u_y, ν, dx, dy, dt, nx, ny)
        u_x_star[1,:] .= 0
        u_x_star[end,:] .= 0

        u_y_star[1,:] .= -u_y_star[2,:]
        u_y_star[end,:] .= -u_y_star[end-1,:]

        u_x_star[:,1] .= -u_x_star[:,2]
        u_x_star[:,end] .= -u_x_star[:,end-1]

        u_y_star[:,1] .= 0
        u_y_star[:,end] .= 0 
        
        @parallel update_div_u_star!(div_u_star, u_x_star, u_y_star, dx, dy, nx, ny)
        div_u_star[1,:] .= -div_u_star[2,:]
        div_u_star[end,:] .= -div_u_star[end-1,:]

        div_u_star[:,1] .= -div_u_star[:,2]
        div_u_star[:,end] .= -div_u_star[:,end-1]

        p .= 0
        dτ = 0.000001
        for i∈1:500
            # boundary conditions ?
            @parallel update_p!(p, p2, div_u_star, ρ, dx, dy, dt, dτ, nx, ny)
            p2[:,1]   .=  p[:,2]     # ∂p/∂x=0 y=0
            p2[:,end] .= -p[:,end-1] # p=0 y=h

            p2[1,:]   .= p[2,:]     # ∂p/∂x=0 x=0
            p2[end,:] .= p[end-1,:] # ∂p/∂x=0 x=w
            
            p .= p2
        end
        
        @parallel update_u!(u_x, u_y, u_x_star, u_y_star, p, ρ, dx, dy, dt, nx, ny)

        u_x[1,:] .= 0
        u_x[end,:] .= 0

        u_y[1,:] .= -u_y[2,:]
        u_y[end,:] .= -u_y[end-1,:]

        u_x[:,1] .= -u_x[:,2]
        u_x[:,end] .= 2*Us-u_x[:,end-1]

        u_y[:,1] .= 0
        u_y[:,end] .= 0         
    end

    u = sqrt.(u_x .^2 .+ u_y .^2)
    
    return u_x, u_y, u, p, nx, ny, dx, dy, w, h
end

u_x, u_y, u, p, nx, ny, dx, dy, w, h = NS_solve()

x_v_domain = (0:(nx-2))*w/(nx-2)
y_v_domain = (0:(ny-2))*h/(ny-2) 
x_v = repeat(x_v_domain', ny-1)
y_v = repeat(y_v_domain, 1, nx-1)

u_x_avg = (u_x[2:end,1:end-1] .+ u_x[2:end,2:end]) ./ 2
u_y_avg = (u_y[1:end-1,2:end] .+ u_y[2:end,2:end]) ./ 2

x_p = (0:(nx-1))*w/(nx-1)
y_p = (0:(ny-1))*h/(ny-1)

PyPlot.contourf(x_p, y_p, p', levels=50)
PyPlot.colorbar()
PyPlot.xlim((0,w))
PyPlot.ylim((0,h))
PyPlot.streamplot(x_v, y_v, u_x_avg', u_y_avg', density=3, color="black", linewidth=0.5, arrowsize=0.5)
PyPlot.savefig("streamlines.pdf", bbox_inches="tight")

