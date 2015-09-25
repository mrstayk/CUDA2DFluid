#include "FluidCUDA.cuh"

// *** Author: Martin Staykov
// *** May, 2015
// ***
// *** References:
// *** Abertay University
// *** rastertek.com
// *** Real-Time Fluid Dynamics for Games, by Jos Stam
// *** NVIDIA CUDA Toolkit
// ***
// *** These functions perform mathematical calculations based on the equations described in my dissertation.
// *** They will not be described in detail here.

#define _(i,j) ((i)+(256)*(j)) // transition from 2d array toa 1d one
#define size 65536 // smoke resolution, total number of cells
#define BLOCKS 90
#define THREADS 736

// The d_ prefix (device) shows this is an array, which lives on the GPU.
float * d_dens, * d_dens_prev, * d_u, * d_v, * d_u_prev, * d_v_prev, * d_aux, * d_MC1, * d_MC2;

__global__ void RedGaussSeidelKernel(int N, float* dest, float* src1, float* src2, float C1, float C2)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	int j = (int) index / (N+2);
	int i = (int) index - (j*(N+2));
	
	if (i<1 || i>N || j<1 || j>N) return;

	if ((i + j)%2 == 0)
	{
		dest[_(i,j)] = (src1[_(i,j)] + C1*(src2[_(i-1,j)] + src2[_(i+1,j)] + src2[_(i,j-1)] + src2[_(i,j+1)]))/C2;
	}
}

__global__ void BlackGaussSeidelKernel(int N, float* dest, float* src1, float* src2, float C1, float C2)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	int j = (int) index / (N+2);
	int i = (int) index - (j*(N+2));

	if (i<1 || i>N || j<1 || j>N) return;

	if ((i + j)%2 != 0)
	{
		dest[_(i,j)] = (src1[_(i,j)] + C1*(src2[_(i-1,j)] + src2[_(i+1,j)] + src2[_(i,j-1)] + src2[_(i,j+1)]))/C2;
	}
}

__global__ void JacobiKernel(int N, float* dest, float* src1, float* src2, float C1, float C2)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	int j = (int) index / (N+2);
	int i = (int) index - (j*(N+2));

	if (i<1 || i>N || j<1 || j>N) return;
	
	dest[_(i,j)] = (src1[_(i,j)] + C1*(src2[_(i-1,j)] + src2[_(i+1,j)] + src2[_(i,j-1)] + src2[_(i,j+1)]))/C2;
}

__global__ void AdvectKernel(int N, float* d, float* d0, float* u, float* v, float dt, bool bBackward)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	int j = (int) index / (N+2);
	int i = (int) index - (j*(N+2));

	int i0, j0, i1, j1;
	float x, y, s0, t0, s1, t1, dt0;

	// Scale the velocity into grid space.
	float dx = 1.0f/N;
	dx = 1/dx;
	dt0 = (dt*dx)/1;

	if (i<1 || i>N || j<1 || j>N) return;

	// The backwards step is used for the MacCormack advection.
	if (bBackward)
	{
		x = i - dt0*u[_(i,j)]; 
		y = j - dt0*v[_(i,j)];
	}
	else
	{
		x = i + dt0*u[_(i,j)]; 
		y = j + dt0*v[_(i,j)];
	}

	if (x<0.5f) x = 0.5f; 
	if (x>N+0.5f) x = N+0.5f; 
	i0 = (int)x; 
	i1 = i0+1;

	if (y<0.5f) y = 0.5f; 
	if (y>N+0.5f) y = N+0.5f; 
	j0 = (int)y; 
	j1 = j0+1;

	s1 = x-i0; 
	s0 = 1-s1; 
	t1 = y-j0; 
	t0 = 1-t1;

	// Interpolate.
	d[_(i,j)] = s0*( t0*d0[_(i0,j0)] + t1*d0[_(i0,j1)] )	+	s1*( t0*d0[_(i1,j0)] + t1*d0[_(i1,j1)] );
}

__global__ void MacCormackKernel(int N, float* dest, float* d0, float* MC1, float* MC2, float* u, float* v, float dt)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	int j = (int) index / (N+2);
	int i = (int) index - (j*(N+2));

	int i0, j0, i1, j1;
	float x, y, s0, t0, s1, t1, dt0;

	// Scale the velocity into grid space.
	float dx = 1.0f/N;
	dx = 1/dx;
	dt0 = dt*dx;

	if (i<1 || i>N || j<1 || j>N) return;
	
	x = i - dt0*u[_(i,j)]; 
	y = j - dt0*v[_(i,j)];

	if (x<0.5f) x = 0.5f; 
	if (x>N+0.5f) x = N+0.5f; 
	i0 = (int)x; 
	i1 = i0+1;

	if (y<0.5f) y = 0.5f; 
	if (y>N+0.5f) y = N+0.5f; 
	j0 = (int)y; 
	j1 = j0+1;
		
	// Get the values of nodes that contribute to the interpolated value.  
	float r0 = d0[_(i0, j0)];
	float r1 = d0[_(i1, j0)];
	float r2 = d0[_(i0, j1)];
	float r3 = d0[_(i1, j1)];		
		
	float result = MC1[_(i,j)] + 0.5f*(d0[_(i,j)] - MC2[_(i,j)]);

	float min = (r0 > r1) ? r1 : r0;
	min = (min > r2) ? r2 : min;
	min = (min > r3) ? r3 : min;

	float max = (r0 < r1) ? r1 : r0;
	max = (max < r2) ? r2 : max;
	max = (max < r3) ? r3 : max;

	// Clamp the result, so that it's stable.
	// If outside the two extrema, revert to results from ordinary advection scheme.
	// The extrema appear to produce errors for unknown reasons. Amend them by adding/subtracting a small number.
	// Too big of a number, and the result produces tearings.
	// Too small and results appear good but blurred, which defeats the purpose of the MacCormack scheme, which is to provide more detail.
	if (result >= (max - 0.02f)) result = MC1[_(i,j)];//max;
	if (result <= (min + 0.02f)) result = MC1[_(i,j)];//min;
		
	dest[_(i,j)] = result;
}

__global__ void DivergenceKernel(int N, float* u, float* v, float* p, float* div)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	int j = (int) index / (N+2);
	int i = (int) index - (j*(N+2));

	if (i<1 || i>N || j<1 || j>N) return;

	float h = 1.0f/N;

	div[_(i,j)] = -0.5f*h*(u[_(i+1,j)] - u[_(i-1,j)] + v[_(i,j+1)] - v[_(i,j-1)]);
	p[_(i,j)] = 0;
}

__global__ void SubtractGradientKernel(int N, float* u, float* v, float* p)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	int j = (int) index / (N+2);
	int i = (int) index - (j*(N+2));

	if (i<1 || i>N || j<1 || j>N) return;

	float h = 1.0f/N;

	u[_(i,j)] -= 0.5f*(p[_(i+1,j)] - p[_(i-1,j)])/h;
	v[_(i,j)] -= 0.5f*(p[_(i,j+1)] - p[_(i,j-1)])/h;
}

__global__ void BoundaryKernel ( int N, int b, float * x)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int i = index + 1;

	if (i<=N)
	{
			x[_(0 ,i)] = b==1 ? -x[_(1,i)] : x[_(1,i)];
			x[_(N+1,i)] = b==1 ? -x[_(N,i)] : x[_(N,i)];
			x[_(i,0 )] = b==2 ? -x[_(i,1)] : x[_(i,1)];
			x[_(i,N+1)] = b==2 ? -x[_(i,N)] : x[_(i,N)];
		
		if (i==1)
		{
			x[_(0 ,0 )] = 0.5f*(x[_(1,0 )]+x[_(0 ,1)]);
			x[_(0 ,N+1)] = 0.5f*(x[_(1,N+1)]+x[_(0 ,N )]);
			x[_(N+1,0 )] = 0.5f*(x[_(N,0 )]+x[_(N+1,1)]);
			x[_(N+1,N+1)] = 0.5f*(x[_(N,N+1)]+x[_(N+1,N )]);
		}
	}
}
  
__global__ void AddSourceKernel(int N, float* x, int i, int j, float value, float dt)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (index != 0) return;
	
	x[_(i, j)] += dt*value;

	x[_(i-1, j)] += dt*value;
	x[_(i+1, j)] += dt*value;

	x[_(i+2, j)] += dt*value;
	x[_(i-2, j)] += dt*value;
}

__global__ void AddSourceMultipleKernel(int N, float * a, float * b, float dt)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	int j = (int) index / (N+2);
	int i = (int) index - (j*(N+2));
	
	if (i<1 || i>N || j<1 || j>N) return;

	a[index] += dt*b[index];
}

__global__ void textureKernel(int N, unsigned char *surface, size_t pitch, float * a)
{
	int y = (int) (threadIdx.x + blockIdx.x * blockDim.x) / (N+2);
	int x = (int) (threadIdx.x + blockIdx.x * blockDim.x) - (y*(N+2));

    float *pixel;

    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't
    // correspond to valid pixels
    if (x >= N+2 || y >= N+2 || (!x) || (!y)) return;
	else
	{
		// get a pointer to the pixel at (x,y)
		pixel = (float *)(surface + y*pitch) + 1*x;

		float pvalue = a[_(x, y)];

		// populate it
		pixel[0] = pvalue;
	}
}

__global__ void BuoyancyKernel(int N, float * dest, float * src, float kappa, float sigma)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	int j = (int) index / (N+2);
	int i = (int) index - (j*(N+2));

	if (i<1 || i>N || j<1 || j>N) return;

	float source = src[_(i, j)];

	dest[_(i, j)] = sigma*source + -kappa*source;
}

__global__ void CurlKernel(int N, float * u, float * v, float * dest, float dt)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	int j = (int) index / (N+2);
	int i = (int) index - (j*(N+2));

	if (i<1 || i>N || j<1 || j>N) return;

	float h = 1.0f/N;//(2*N);
	float du_dy;
	float dv_dx;

	du_dy = (u[_(i, j+1)] - u[_(i, j-1)]) / h * 0.5f;
    dv_dx = (v[_(i+1, j)] - v[_(i-1, j)]) / h * 0.5f;

    dest[_(i, j)] = (dv_dx - du_dy);// * h * 0.5f;
}

__global__ void VorticityKernel(int N, float * u, float * v, float * c, float dt, float vort)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	int j = (int) index / (N+2);
	int i = (int) index - (j*(N+2));

	if (i<1 || i>N || j<1 || j>N) return;

	float h = 1.0f/N;//(2*N);

	float omegaT = (c[_(i, j-1)]);
	float omegaB = (c[_(i, j+1)]);
	float omegaR = (c[_(i+1, j)]);
	float omegaL = (c[_(i-1, j)]);

	float comp1 = omegaT - omegaB;
	float comp2 = omegaR - omegaL;

	float2 force; force.x = comp1; force.y = comp2; force *= 0.5f; force /= h;
	
	force /= (length(force) + 0.00001f);

	float2 NN;
	NN.x = -c[_(i, j)]*force.y;
	NN.y = c[_(i, j)]*force.x;

	NN *= vort;

	u[_(i, j)] += NN.x * dt;
	v[_(i, j)] += NN.y * dt;
}

void InitCUDA()
{
	// Allocate all arrays on the device.
	if (cudaMalloc((void**)&d_u, size * sizeof(float)) != cudaSuccess)
	{
		return;
	}

	if (cudaMalloc((void**)&d_u_prev, size * sizeof(float)) != cudaSuccess)
	{
		cudaFree(d_u);
		return;
	}
	
	if (cudaMalloc((void**)&d_v, size * sizeof(float)) != cudaSuccess)
	{
		cudaFree(d_u);
		cudaFree(d_u_prev);
		return;
	}

	if (cudaMalloc((void**)&d_v_prev, size * sizeof(float)) != cudaSuccess)
	{
		cudaFree(d_u);
		cudaFree(d_u_prev);
		cudaFree(d_v);
		return;
	}

	if (cudaMalloc((void**)&d_dens, size * sizeof(float)) != cudaSuccess)
	{
		cudaFree(d_u);
		cudaFree(d_u_prev);
		cudaFree(d_v);
		cudaFree(d_v_prev);
		return;
	}

	if (cudaMalloc((void**)&d_dens_prev, size * sizeof(float)) != cudaSuccess)
	{
		cudaFree(d_u);
		cudaFree(d_u_prev);
		cudaFree(d_v);
		cudaFree(d_v_prev);
		cudaFree(d_dens);
		return;
	}

	if (cudaMalloc((void**)&d_aux, size * sizeof(float)) != cudaSuccess)
	{
		cudaFree(d_u);
		cudaFree(d_u_prev);
		cudaFree(d_v);
		cudaFree(d_v_prev);
		cudaFree(d_dens);
		cudaFree(d_dens_prev);
		return;
	}

	if (cudaMalloc((void**)&d_MC1, size * sizeof(float)) != cudaSuccess)
	{
		cudaFree(d_u);
		cudaFree(d_u_prev);
		cudaFree(d_v);
		cudaFree(d_v_prev);
		cudaFree(d_dens);
		cudaFree(d_dens_prev);
		cudaFree(d_aux);
		return;
	}

	if (cudaMalloc((void**)&d_MC2, size * sizeof(float)) != cudaSuccess)
	{
		cudaFree(d_u);
		cudaFree(d_u_prev);
		cudaFree(d_v);
		cudaFree(d_v_prev);
		cudaFree(d_dens);
		cudaFree(d_dens_prev);
		cudaFree(d_aux);
		cudaFree(d_MC1);
		return;
	}

	// Initialize the arrays to 0.
	if (cudaMemset(d_dens, 0, size * sizeof(float)) != cudaSuccess)
	{
		FreeCUDA();
		return;
	}
	if (cudaMemset(d_dens_prev, 0, size * sizeof(float)) != cudaSuccess)
	{
		FreeCUDA();
		return;
	}
	if (cudaMemset(d_u, 0, size * sizeof(float)) != cudaSuccess)
	{
		FreeCUDA();
		return;
	}
	if (cudaMemset(d_u_prev, 0, size * sizeof(float)) != cudaSuccess)
	{
		FreeCUDA();
		return;
	}
	if (cudaMemset(d_v, 0, size * sizeof(float)) != cudaSuccess)
	{
		FreeCUDA();
		return;
	}
	if (cudaMemset(d_v_prev, 0, size * sizeof(float)) != cudaSuccess)
	{
		FreeCUDA();
		return;
	}
	if (cudaMemset(d_aux, 0, size * sizeof(float)) != cudaSuccess)
	{
		FreeCUDA();
		return;
	}
	if (cudaMemset(d_MC1, 0, size * sizeof(float)) != cudaSuccess)
	{
		FreeCUDA();
		return;
	}
	if (cudaMemset(d_MC2, 0, size * sizeof(float)) != cudaSuccess)
	{
		FreeCUDA();
		return;
	}

	// Exit and return void.
	return;
}

void FreeCUDA()
{
	cudaFree(d_dens);
	cudaFree(d_dens_prev);
	cudaFree(d_u);
	cudaFree(d_u_prev);
	cudaFree(d_v);
	cudaFree(d_v_prev);
	cudaFree(d_aux);
	cudaFree(d_MC1);
	cudaFree(d_MC2);

	cudaDeviceReset();
}

void DiffuseCUDA(int N, float b, float * x, float * x0, float diff, float dt, int iterations, bool isJacobi)
{
	float a = dt*diff*(float)N*(float)N;

	for (int k=0; k<iterations; k++) 
	{		
		if (isJacobi) 
		{
			JacobiKernel<<<BLOCKS, THREADS>>>(N, d_aux, x0, x, a, 1+4*a);
			cudaDeviceSynchronize();
			x = d_aux;
		}
		else 
		{
			RedGaussSeidelKernel<<<BLOCKS, THREADS>>>(N, x, x0, x, a, 1+4*a);
			cudaDeviceSynchronize();
			BlackGaussSeidelKernel<<<BLOCKS, THREADS>>>(N, x, x0, x, a, 1+4*a);
		}

		cudaDeviceSynchronize();
		BoundaryKernel<<<1, N>>>(N, b, x);
		cudaDeviceSynchronize();
	}
}

void AdvectCUDA(int N, int b, float* d, float* d0, float* u, float* v, float dt, bool bBackward)
{
	AdvectKernel<<<BLOCKS, THREADS>>>(N, d, d0, u, v, dt, bBackward);
	cudaDeviceSynchronize();
	BoundaryKernel<<<1, N>>>(N, b, d);
	cudaDeviceSynchronize();
}

void AdvectMacCormackCUDA(int N, int b, float* d, float* d0, float* u, float* v, float dt)
{
	AdvectKernel<<<BLOCKS, THREADS>>>(N, d_MC1, d0, u, v, dt, true);
	cudaDeviceSynchronize();
	BoundaryKernel<<<1, N>>>(N, b, d_MC1);
	cudaDeviceSynchronize();

	AdvectKernel<<<BLOCKS, THREADS>>>(N, d_MC2, d_MC1, u, v, dt, false);
	cudaDeviceSynchronize();
	BoundaryKernel<<<1, N>>>(N, b, d_MC2);
	cudaDeviceSynchronize();

	MacCormackKernel<<<BLOCKS, THREADS>>>(N, d, d0, d_MC1, d_MC2, u, v, dt);
	cudaDeviceSynchronize();
	BoundaryKernel<<<1, N>>>(N, b, d);
	cudaDeviceSynchronize();
}

void ProjectCUDA(int N, float* u, float* v, float* p, float* div, int iterations, bool isJacobi)
{
	DivergenceKernel<<<BLOCKS, THREADS>>>(N, u, v, p, div);
	cudaDeviceSynchronize();
	BoundaryKernel<<<1, N>>>(N, 0, div);
	BoundaryKernel<<<1, N>>>(N, 0, p);
	cudaDeviceSynchronize();
	
	for ( int k=0; k<iterations; k++ ) 
	{
		if (isJacobi) 
		{
			JacobiKernel<<<BLOCKS, THREADS>>>(N, d_aux, div, p, 1, 4);
			cudaDeviceSynchronize();
			p = d_aux;
		}
		else 
		{
			RedGaussSeidelKernel<<<BLOCKS, THREADS>>>(N, p, div, p, 1, 4);
			cudaDeviceSynchronize();
			BlackGaussSeidelKernel<<<BLOCKS, THREADS>>>(N, p, div, p, 1, 4);
		}

		cudaDeviceSynchronize();
		BoundaryKernel<<<1, N>>>(N, 0, p);
		cudaDeviceSynchronize();
	}

	SubtractGradientKernel<<<BLOCKS, THREADS>>>(N, u, v, p);
	cudaDeviceSynchronize();
	BoundaryKernel<<<1, N>>>(N, 1, u);
	BoundaryKernel<<<1, N>>>(N, 2, v);
	cudaDeviceSynchronize();
}

void FrameCUDA(int N, int source_vel_i, int source_vel_j, int source_dens_i, int source_dens_j, float source_u_value, float source_v_value, float source_dens_value, Parameters a)
{
	float * tempPtr; // used for swapping arrays
	
	// Velocity step.
	AddSourceKernel<<<1, 1>>>(N, d_u, source_vel_i, source_vel_j, source_u_value, a.dt); 
	AddSourceKernel<<<1, 1>>>(N, d_v, source_vel_i, source_vel_j, source_v_value, a.dt);
	cudaDeviceSynchronize();		

	if (a.bViscosity)
	{
		if (a.enumIterativeMethod == Jacobi) DiffuseCUDA(N, 1, d_u_prev, d_u, a.fViscFactor, a.dt, a.iPoissonIterations, true);
		else if (a.enumIterativeMethod == GaussSeidel) DiffuseCUDA(N, 1, d_u_prev, d_u, a.fViscFactor, a.dt, a.iPoissonIterations, false);
		if (a.enumIterativeMethod == Jacobi) DiffuseCUDA(N, 2, d_v_prev, d_v, a.fViscFactor, a.dt, a.iPoissonIterations, true);
		else if (a.enumIterativeMethod == GaussSeidel) DiffuseCUDA(N, 2, d_v_prev, d_v, a.fViscFactor, a.dt, a.iPoissonIterations, false);
		tempPtr = d_u; d_u = d_u_prev; d_u_prev = tempPtr;
		tempPtr = d_v; d_v = d_v_prev; d_v_prev = tempPtr;
		if (a.enumIterativeMethod == Jacobi) ProjectCUDA(N, d_u, d_v, d_u_prev, d_v_prev, a.iPoissonIterations, true);
		else if (a.enumIterativeMethod == GaussSeidel) ProjectCUDA(N, d_u, d_v, d_u_prev, d_v_prev, a.iPoissonIterations, false);
	}

	if (a.enumCurrentAdvect == SingleSL)
	{
		AdvectCUDA(N, 1, d_u_prev, d_u, d_u, d_v, a.dt, true); 
		AdvectCUDA(N, 2, d_v_prev, d_v, d_u, d_v, a.dt, true);
	}
	else if (a.enumCurrentAdvect == SemiMacCormack)
	{
		AdvectMacCormackCUDA(N, 1, d_u_prev, d_u, d_u, d_v, a.dt); 
		AdvectMacCormackCUDA(N, 2, d_v_prev, d_v, d_u, d_v, a.dt);
	}

	if (a.bVorticity)
	{
		// *** VORTICITY CONFINEMENT ***
		CurlKernel<<<BLOCKS, THREADS>>>(N, d_u_prev, d_v_prev, d_aux, a.dt);
		cudaDeviceSynchronize();
		VorticityKernel<<<BLOCKS, THREADS>>>(N, d_u_prev, d_v_prev, d_aux, a.dt, a.fVortStrength);
		cudaDeviceSynchronize();
		BoundaryKernel<<<1, N>>>(N, 1, d_u_prev);
		BoundaryKernel<<<1, N>>>(N, 2, d_v_prev);
		cudaDeviceSynchronize();
	}

	if (a.bBuoyancy)
	{
		BuoyancyKernel<<<BLOCKS, THREADS>>>(N, d_v, d_dens, a.fKappa, a.fSigma);
		cudaDeviceSynchronize();
		AddSourceMultipleKernel<<<BLOCKS, THREADS>>>(N, d_v_prev, d_v, a.dt);
		cudaDeviceSynchronize();
		BoundaryKernel<<<1, N>>>(N, 2, d_v_prev);
		cudaDeviceSynchronize();
	}
		
	if (a.enumIterativeMethod == Jacobi) ProjectCUDA(N, d_u_prev, d_v_prev, d_u, d_v, a.iPoissonIterations, true);
	else if (a.enumIterativeMethod == GaussSeidel)  ProjectCUDA(N, d_u_prev, d_v_prev, d_u, d_v, a.iPoissonIterations, false);

	tempPtr = d_u; d_u = d_u_prev; d_u_prev = tempPtr;
	tempPtr = d_v; d_v = d_v_prev; d_v_prev = tempPtr;

	tempPtr = d_dens; d_dens = d_dens_prev; d_dens_prev = tempPtr;

	// Density step.
	AddSourceKernel<<<1, 1>>>(N, d_dens_prev, source_dens_i, source_dens_j, source_dens_value, a.dt);
	AddSourceKernel<<<1, 1>>>(N, d_dens_prev, 128, 248, 50, a.dt);
	cudaDeviceSynchronize();
	if (a.bDiffusion)
	{
		if (a.enumIterativeMethod == Jacobi) DiffuseCUDA(N, 0, d_dens, d_dens_prev, a.fDiffFactor, a.dt, a.iPoissonIterations, true);
		else if (a.enumIterativeMethod == GaussSeidel) DiffuseCUDA(N, 0, d_dens, d_dens_prev, a.fDiffFactor, a.dt, a.iPoissonIterations, false);
		tempPtr = d_dens; d_dens = d_dens_prev; d_dens_prev = tempPtr;
	}	
	if (a.enumCurrentAdvect == SingleSL) AdvectCUDA(N, 0, d_dens, d_dens_prev, d_u, d_v, a.dt, true);
	else if (a.enumCurrentAdvect == SemiMacCormack) AdvectMacCormackCUDA(N, 0, d_dens, d_dens_prev, d_u, d_v, a.dt);
	
	return;
}

void cuda_texture_2d(int N, void *surface, int width, int height, size_t pitch)
{
    textureKernel<<<BLOCKS, THREADS>>>(N, (unsigned char *)surface, pitch, d_dens);
}