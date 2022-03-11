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

macro dxc(A, ix, iy, iz) (A[ix+1,iy,iz]-A[ix-1,iy,iz]) end
macro dyc(A, ix, iy, iz) (A[ix,iy+1,iz]-A[ix,iy-1,iz]) end
macro dzc(A, ix, iy, iz) (A[ix,iy,iz+1]-A[ix,iy,iz-1]) end

macro dx(A, ix, iy, iz) (A[ix+1,iy,iz]-A[ix,iy,iz]) end
macro dy(A, ix, iy, iz) (A[ix,iy+1,iz]-A[ix,iy,iz]) end
macro dz(A, ix, iy, iz) (A[ix,iy,iz+1]-A[ix,iy,iz]) end

macro dx2(A, ix, iy, iz) (A[ix+1,iy,iz]+A[ix-1,iy,iz]-2A[ix,iy,iz]) end
macro dy2(A, ix, iy, iz) (A[ix,iy+1,iz]+A[ix,iy-1,iz]-2A[ix,iy,iz]) end
macro dz2(A, ix, iy, iz) (A[ix,iy,iz+1]+A[ix,iy,iz-1]-2A[ix,iy,iz]) end

macro Δ(A, ix, iy, iz) A[ix+1,iy,iz]+A[ix-1,iy,iz]+A[ix,iy+1,iz]+A[ix,iy-1,iz]+A[ix,iy,iz+1]+A[ix,iy,iz-1]-6A[ix,iy,iz]) end

macro id(A, ix, iy, iz) A[ix,iy,iz] end
macro avg_xy(A, ix, iy, iz) (A[ix+1,iy+1,iz]+A[ix+1,iy,iz]+A[ix,iy+1,iz]+A[ix,iy,iz]) end
macro avg_xz(A, ix, iy, iz) (A[ix+1,iy,iz+1]+A[ix+1,iy,iz]+A[ix,iy,iz+1]+A[ix,iy,iz]) end
macro avg_yz(A, ix, iy, iz) (A[ix,iy+1,iz+1]+A[ix,iy+1,iz]+A[ix,iy,iz+1]+A[ix,iy,iz]) end

function ns_step!(u_x, u_y, u_z, u_x2, u_y2, y_z2, div_u_star, p, p2, ν, rho, dt, dx, dy, dz)

    @parallel update_ustar!(u_x, u_y, u_z, u_x2, u_y2, u_z2, u_xstar, u_ystar, u_zstar, ν, dx, dy, dz, dt)
    # boundary conditions on u_star ?

    @parallel compute_div_u_star!(u_starx, u_stary, u_starz, div_u_star, dx, dy, dz)
    # boundary conditions on div_u_star ?
    
    # initialize p
    while norm(p-p2) > ε
        @parallel update_pressure!(div_u_star, dτ, dt, rho, p, p2)

        @parallel neumann_bc_x!(p2)
        @parallel neumann_bc_y!(p2)
        @parallel neumann_bc_z!(p2)
        
        p, p2 = p2, p
    end

    @parallel update_u!(u_x, u_y, u_z, u_x2, u_y2, u_z2, p, rho, dt)
    
    @parallel dirichlet_bc_x!(u_x2, 0, 0)
    @parallel dirichlet_bc_x!(u_y2, 0, 0)
    @parellel dirichlet_bc_x!(u_z2, 0, 0)

    @parallel dirichlet_bc_y!(u_x2, 0, 0)
    @parallel dirichlet_bc_y!(u_y2, 0, 0)
    @parallel dirichlet_bc_y!(u_z2, 0, 0)

    @parallel dirichlet_bc_z!(u_x2, 0, u)
    @parallel dirichlet_bc_z!(u_y2, 0, 0)
    @parallel dirichlet_bc_z!(u_z2, 0, 0)    

end

@parallel_indices (ix, iy, iz) function neumann_bc_x!(A)
    A[1,iy,iz] = A[2,iy,iz]
    A[end,iy,iz] A[end-1,iy,iz]
end

@parallel_indices (ix, iy, iz) function neumann_bc_y!(A)
    A[ix,1,iz] = A[ix,2,iz]
    A[ix,end,iz] = A[ix,end-1,iz]
end

@parallel_indices (ix, iy, iz) function neumann_bc_z!(A)
    A[ix,iy,1] = A[ix,iy,2]
    A[ix,iy,end] = A[ix,iy,end-1]
end

@parallel_indices (ix, iy, iz) function dirichlet_bc_x!(A, l, h)
    A[1,iy,iz] = l
    A[end,iy,iz] = h
end

@parallel_indices (ix, iy, iz) function dirichlet_bc_y!(A, l, h)
    A[ix,1,iz] = l
    A[ix,end,iz] = h
end

@parallel_indices (ix, iy, iz) function dirichlet_bc_z!(A, l, h)
    A[ix,iy,1] = l
    A[ix,iy,end] = h
end
          

@parallel_indices (ix, iy, iz) function update_ustar!(u_x, u_y, u_z, u_xstar, u_ystar, u_zstar, ν, dx, dy, dz, dt)
    if ix>1 && ix<size(u_x,1) && iy>1 && iy<size(u_x,2) && iz>1 && iz<size(u_x,3)
        U_xstar[ix,iy,iz] = u_x[ix,iy,iz] + dt*(ν*(@dx2(u_x,ix,iy,iz)/dx^2+@dy2(u_x,ix,iy,iz)/dy^2+@dz2(u_x,ix,iy,iz)/dz^2) - ( @id(u_x,ix,iy,iz)*@dxc(u_x,ix,iy,iz)/2dx + @avg(u_y,ix,iy,iz)*@dyc(u_x,ix,iy,iz)/2dy + @avg(u_z)*@dz2(u_x,ix,iy,iz)/2dz))
        u_ystar[ix,iy,iz] = u_y[ix,iy,iz] + dt*(ν*(@dx2(u_y,ix,iy,iz)/dx^2+@dy2(u_y,ix,iy,iz)/dy^2+@dz2(u_y,ix,iy,iz)/dz^2) - (@avg(u_x,ix,iy,iz)*@dxc(u_y,ix,iy,iz)/2dx +  @id(u_y,ix,iy,iz)*@dyc(u_y,ix,iy,iz)/2dy + @avg(u_z)*@dz2(u_y,ix,iy,iz)/2dz))
        u_zstar[ix,iy,iz] = u_z[ix,iy,iz] + dt*(ν*(@dx2(u_z,ix,iy,iz)/dx^2+@dy2(u_z,ix,iy,iz)/dy^2+@dz2(u_z,ix,iy,iz)/dz^2) - (@avg(u_x,ix,iy,iz)*@dxc(u_z,ix,iy,iz)/2dx + @avg(u_y,ix,iy,iz)*@dyc(u_z,ix,iy,iz)/2dy +  @id(u_z)*@dz2(u_z,ix,iy,iz)/2dz))
        
    end
end

@parallel_indices (ix, iy, iz) function compute_div_u_star(u_starx, u_stary, u_starz, div_u_star, dx, dy, dz)
    if ix>1 && ix<size(u_x,1) && iy>1 && iy<size(u_x,2) && iz>1 && iz<size(u_x,3)
        div_u_star[ix, iy, iz] = @dx(u_starx,ix,iy,iz)/dx + @dy(u_stary,ix,iy,iz)/dy + @dz(u_starz,ix,iy,iz)/dz
    end
end

@parallel_indices (ix, iy, iz) function update_pressure!(div_u_star, dτ, dt, rho, p, p2)
    if ix>1 && ix<size(u_x,1) && iy>1 && iy<size(u_x,2) && iz>1 && iz<size(u_x,3)
        p2[ix,iy,iz] = p[ix,iy,iz] -dτ*(@dx2(p,ix,iy,iz)/dx^2+@dy2(p,ix,iy,iz)/dy^2+@dz2(p,ix,iy,iz)/dz^2 + rho/dt*div_u_star[ix,iy,iz])
    end
end

@parallel_indices (ix, iy, iz) function update_u!(u_x, u_y, u_z, u_x2, u_y2, u_z2, p, rho, dt)
    if ix>1 && ix<size(u_x,1) && iy>1 && iy<size(u_x,2) && iz>1 && iz<size(u_x,3)
        u_x2[ix,iy,iz] = u_starx[ix,iy,iz] - dt/rho*@dx(p,ix,iy,iz)/dx
        u_y2[ix,iy,iz] = u_stary[ix,iy,iz] - dt/rho*@dy(p,ix,iy,iz)/dy
        u_z2[ix,iy,iz] = u_starz[ix,iy,iz] - dt/rho*@dz(p,ix,iy,iz)/dz        
    end
end

function ns3D()
# Physics
ρ = 1.0;
ν = 1.0;
lx, ly, lz = 1.0, 1.0, 1.0;

# Numerics
nx, ny, nz = 64, 64, 64;
nt         = 100;
dx         = lx/(nx-1);
dy         = ly/(ny-1);
dz         = lz/(nz-1);

# Array initializations
u_x = @zeros(nx, ny, nz);
u_y = @zeros(nx, ny, nz);
u_z = @zeros(nx, ny, nz);
div_u_star = @zeros(nx, ny, nz); 
p = @zeros(nx, ny, nz);
    
u_xstar = @zeros(nx, ny, nz);
u_ystar = @zeros(nx, ny, nz);
u_zstar = @zeros(nx, ny, nz);
p2 = @zeros(nx, ny, nz);

# Time loop
dt = min(dx^2,dy^2,dz^2);
for it = 1:nt
    # @parallel
    ns_step!(u_x, u_y, u_z, u_x2, u_y2, u_z2, div_u_star, p, p2, ν, rho, dt, dx, dy, dz);
    u_x, u_x2 = u_x2, u_x
    u_y, u_y2 = u_y2, u_y
    u_z, u_z2 = u_z2, u_z    
end

return u_x, u_y, u_z
    
end

u_x, u_y, u_z = ns3D()
