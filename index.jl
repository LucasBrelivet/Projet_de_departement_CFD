const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using LinearAlgebra
using Plots

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);
end

macro dx(A) A[3:end,2:end-1,2:end-1]-A[1:end-2,2:end-1,2:end-1] end
macro dy(A) A[2:end-1,3:end,2:end-1]-A[2:end-1,1:end-2,2:end-1] end
macro dz(A) A[2:end-1,2:end-1,3:end]-A[2:end-1,2:end-1,1:end-2] end

macro moy_x(A) 1/8*(A[1:end-2,2:end-1,2:end-1] + A[2:end-1,2:end-1,2:end-1] + A[1:end-2,3:end,2:end-1] + A[2:end-1,3:end,2:end-1] + A[1:end-2,2:end-1,3:end] + A[2:end-1,2:end-1,3:end] + A[1:end-2,3:end,3:end] + A[2:end-1,3:end,3:end]) end

macro moy_y(A) 1/8*(A[2:end-1,1:end-2,2:end-1] + A[2:end-1,2:end-1,2:end-1] + A[3:end,1:end-2,2:end-1] + A[3:end,2:end-1,2:end-1] + A[2:end-1,1:end-2,3:end] + A[2:end-1,2:end-1,3:end] + A[3:end,1:end-2,3:end] + A[3:end,2:end-1,3:end]) end

macro moy_z(A) 1/8*(A[2:end-1,2:end-1,1:end-2] + A[2:end-1,2:end-1,2:end-1] + A[3:end,2:end-1,1:end-2] + A[3:end,2:end-1,2:end-1] + A[2:end-1,3:end,1:end-2] + A[2:end-1,3:end,2:end-1] + A[3:end,3:end,1:end-2] + A[3:end,3:end,2:end-1]) end


@parallel function ns_step!(u_x, u_y, u_z, u_x2, u_y2, y_z2, div_u, p, p2, ν, rho, dt, dx, dy, dz)
    
    @inn(u_x2) = @inn(u_x) + dt*(ν*(@d2_x(u_x)/(dx^2) + @d2_y(u_x)/(dy^2) + @d2_z(u_x)/(dz^2)) - (  @inn(u_x).*@dx(u_x)/(2dx) + @moy_x(u_y).*@dy(u_x)/(2dy) + @moy_x(u_z).*@dz(u_x)/(2dz)))
    @inn(u_y2) = @inn(u_y) + dt*(ν*(@d2_x(u_y)/(dx^2) + @d2_y(u_y)/(dy^2) + @d2_z(u_y)/(dz^2)) - (@moy_y(u_x).*@dx(u_y)/(2dx) + @inn(u_y).*@dy(u_y)/(2dy)   + @moy_y(u_z).*@dz(u_y)/(2dz)))
    @inn(u_z2) = @inn(u_z) + dt*(ν*(@d2_x(u_z)/(dx^2) + @d2_y(u_z)/(dy^2) + @d2_z(u_z)/(dz^2)) - (@moy_z(u_x).*@dx(u_z)/(2dx) + @moy_z(u_y).*@dy(u_z)/(2dy) + @inn(u_z).*@dz(u_z)/(2dz)))       
    
    div_u = @d_xi(u_x2)/dx + @d_yi(u_y2)/dy + @d_zi(u_z2)/dz

    I = [CartesianIndex(0,0,0),
         CartesianIndex(-1,0,0),
         CartesianIndex( 1,0,0),
         CartesianIndex(0,-1,0),
         CartesianIndex(0,1 ,0),
         CartesianIndex(0,0,-1),
         CartesianIndex(0,0, 1)]
    Δ = [-6, 1, 1, 1, 1, 1, 1]
    p .= 0
    Δτ = 0.01
    while norm(p-p2) > ε

        for i∈@inn(CartesianIndices(p))
            p2[i] += -Δτ*(sum(Δ.*p[I.+i]) + rho/dt*div_u[i])
        end
        # zero derivative boundary condition
        p2[1,2:end-1,2:end-1] .= p2[2,2:end-1,2:end-1]
        p2[end,2:end-1,2:end-1] .= p2[end-1,2:end-1,2:end-1]

        p2[2:end-1,1,2:end-1] = p2[2:end-1,2,2:end-1]
        p2[2:end-1,end,2:end-1] = p2[2:end-1,end-1,2:end-1]

        p2[2:end-1,2:end-1,1] = p2[2:end-1,2:end-1,2]
        p2[2:end-1,2:end-1,end] = p2[2:end-1,2:end-1,end-1]
        p, p2 = p2, p
    end

    @inn(u_x) = @inn(u_x2) - dt/rho*(p[2:end-1,2:end-1,2:end-1]-p[1:end-3,2:end-1,2:end-1])
    @inn(u_y) = @inn(u_y2) - dt/rho*(p[2:end-1,2:end-1,2:end-1]-p[2:end-1,1:end-3,2:end-1])
    @inn(u_z) = @inn(u_z2) - dt/rho*(p[2:end-1,2:end-1,2:end-1]-p[2:end-1,2:end-1,3:end-1])

    # Boundary conditions
    return
end

function ns3D()
# Physics
ρ = 1.0;
ν = 1.0;
lx, ly, lz = 1.0, 1.0, 1.0;

# Numerics
nx, ny, nz = 64, 64, 64;                              # Number of gridpoints in dimensions x, y and z
nt         = 100;                                        # Number of time steps
dx         = lx/(nx-1);                                  # Space step in x-dimension
dy         = ly/(ny-1);                                  # Space step in y-dimension
dz         = lz/(nz-1);                                  # Space step in z-dimension

# Array initializations
u_x = @zeros(nx, ny, nz);
u_y = @zeros(nx, ny, nz);
u_z = @zeros(nx, ny, nz);
div_u = @zeros(nx, ny, nz); 
p = @zeros(nx, ny, nz);
    
u_x2 = @zeros(nx, ny, nz);
u_y2 = @zeros(nx, ny, nz);
u_z2 = @zeros(nx, ny, nz);
p2 = @zeros(nx, ny, nz);

# Time loop
dt = min(dx^2,dy^2,dz^2);
for it = 1:nt
    @parallel ns_step!(u_x, u_y, u_z, u_x2, u_y2, u_z2, div_u, p, p2, ν, rho, dt, dx, dy, dz);
    u_x, u_x2 = u_x2, u_x
    u_y, u_y2 = u_y2, u_y
    u_z, u_z2 = u_z2, u_z    
end

return u_x, u_y, u_z
    
end

u_x, u_y, u_z = ns3D()
