#ifndef _FLUIDCUDA_CU_
#define _FLUIDCUDA_CU_

// *** Author: Martin Staykov, 2015
// *** References:
// *** Abertay University
// *** rastertek.com
// *** Real-Time Fluid Dynamics for Games, by Jos Stam
// *** NVIDIA CUDA Toolkit
// ***
// *** This is where all the actual GPU physics logic takes place.

#include <windows.h>
#include <mmsystem.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <dynlink_d3d11.h>
#include <iostream>
#include <cuda.h>
#include <device_functions.h>

#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>

#include <rendercheck_d3d11.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <helper_math.h>

using namespace std;

typedef enum { SemiMacCormack=1, SingleSL } Advection;
typedef enum { GaussSeidel=1, Jacobi } IterativeMethod;

struct Parameters
{
	Advection enumCurrentAdvect;
	IterativeMethod enumIterativeMethod;
	float fVortStrength;
	float fKappa;
	float fSigma;	
	int iPoissonIterations;
	bool bDiffusion;
	bool bVorticity;
	bool bBuoyancy;
	bool bViscosity;
	float dt;
	float fViscFactor;
	float fDiffFactor;
	float * color;
};

// Kernels, executed by many threads on the GPU.
__global__ void AdvectKernel(int N, float* d, float* d0, float* u, float* v, float dt, bool bBackward);
__global__ void DivergenceKernel(int N, float* u, float* v, float* p, float* div);
__global__ void SubtractGradientKernel(int N, float* u, float* v, float* p);
__global__ void BoundaryKernel(int N, int b, float * x);
__global__ void AddSourceKernel(int N, float* x, int i, int j, float dt);
__global__ void AddSourceMultipleKernel(int N, float * a, float * b, float dt);
__global__ void textureKernel(int N, unsigned char *surface, size_t pitch, float * a);
__global__ void BuoyancyKernel(int N, float * dest, float * src, float kappa, float sigma);
__global__ void CurlKernel(int N, float * a, float * b, float * dest, float dt);
__global__ void VorticityKernel(int N, float * a, float * b, float * c, float dt, float vort);
__global__ void MacCormackKernel(int N, float* dest, float* d0, float* MC1, float* MC2, float* u, float* v, float dt);
__global__ void RedGaussSeidelKernel(int N, float* dest, float* src1, float* src2, float C1, float C2);
__global__ void BlackGaussSeidelKernel(int N, float* dest, float* src1, float* src2, float C1, float C2);
__global__ void JacobiKernel(int N, float* dest, float* src1, float* src2, float C1, float C2);

// Functions, executed on the CPU. They invoke GPU functions.
void InitCUDA();
void FreeCUDA();
void FrameCUDA(int N, int source_vel_i, int source_vel_j, int source_dens_i, int source_dens_j, float source_u_value, float source_v_value, float source_dens_value, Parameters a);

void DiffuseCUDA(int N, float b, float * x, float * x0, float diff, float dt, int iterations, bool isJacobi);
void AdvectCUDA(int N, int b, float* d, float* d0, float* u, float* v, float dt, bool bBackward);
void AdvectMacCormackCUDA(int N, int b, float* d, float* d0, float* u, float* v, float dt);
void ProjectCUDA(int N, float* u, float* v, float* p, float* div, int iterations, bool isJacobi);
void AddSourceCUDA();
void cuda_texture_2d(int N, void *surface, int width, int height, size_t pitch);

#endif // #ifndef _FLUIDCUDA_CU_