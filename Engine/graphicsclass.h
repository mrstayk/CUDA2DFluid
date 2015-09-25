#ifndef _GRAPHICSCLASS_H_
#define _GRAPHICSCLASS_H_

// *** Author: Martin Staykov
// *** May, 2015
// ***
// *** References:
// *** Abertay University
// *** rastertek.com
// *** NVIDIA CUDA Toolkit

#include "d3dclass.h"
#include "cameraclass.h"
#include "modelclass.h"
#include "Shader Classes\textureshaderclass.h"
#include "inputclass.h"
#include "textclass.h"
#include "fpsclass.h"

#include "CUDA Source Files\FluidCUDA.cuh"
#include <AntTweakBar.h>

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>

#define IX(i,j) ((i)+(N+2)*(j)) // transition from 2d array to a 1d one
#define size1 65536 // smoke resolution, total number of cells

const bool FULL_SCREEN = false;
const bool VSYNC_ENABLED = true;
const float SCREEN_DEPTH = 1000.0f;
const float SCREEN_NEAR = 0.1f;

class GraphicsClass
{
private:
	struct
	{
		ID3D11Texture2D				*pTexture;
		ID3D11ShaderResourceView	*pSRView;
		cudaGraphicsResource		*cudaResource;
		void						*cudaLinearMemory;
		size_t						pitch;
		int							width;
		int							height;
	} g_texture_2d; // the smoke is a 2d texture

public:
	GraphicsClass();
	GraphicsClass(const GraphicsClass&);
	~GraphicsClass();

	bool Initialize(HINSTANCE, HWND, int, int);
	void Shutdown();
	bool Frame();

private:
	bool Render();
	bool HandleInput();

private:
	D3DClass* m_D3D;
	CameraClass* m_Camera;
	ModelClass* m_Model;
	TextureShaderClass* m_TextureShader;
	InputClass * m_Input;
	TextClass* m_Text;
	FontShaderClass* m_FontShader;
	FpsClass* m_Fps;

	int omx, omy; // old mouse coords
	int source_vel_i, source_vel_j;
	int source_dens_i, source_dens_j;
	float source_u_value, source_v_value, source_dens_value;

	bool scenePaused;
	Parameters PARAMS; // app parameters
};

#endif