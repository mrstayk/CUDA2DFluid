// *** Author: Martin Staykov
// *** May, 2015
// ***
// *** References:
// *** Abertay University
// *** rastertek.com
// *** NVIDIA CUDA Toolkit

#include "graphicsclass.h"

GraphicsClass::GraphicsClass()
{
	m_D3D = 0;
	m_Camera = 0;
	m_Model = 0;
	m_TextureShader = 0;
	m_Input = 0;
	m_Text = 0;
	m_FontShader = 0;
	m_Fps = 0;
	
	omx = omy = 0;

	source_vel_i = 0;
	source_vel_j = 0;
	source_dens_i = 0;
	source_dens_j = 0;
	source_u_value = 0;
	source_v_value = 0;
	source_dens_value = 0;

	scenePaused = false;
}


GraphicsClass::GraphicsClass(const GraphicsClass& other)
{
}


GraphicsClass::~GraphicsClass()
{
}

bool GraphicsClass::Initialize(HINSTANCE hinstance, HWND hwnd, int screenWidth, int screenHeight)
{
	if (!dynlinkLoadD3D11API())                  // Search for D3D API (locate drivers, does not mean device is found)
    {
        printf("> D3D11 API libraries NOT found on.. Exiting.\n");
        dynlinkUnloadD3D11API();
        return false;
    }

	bool result;
	D3DXMATRIX baseViewMatrix;
	char videoCard[128];
	int videoMemory;

	// Create the Direct3D object.
	m_D3D = new D3DClass;
	if(!m_D3D)
	{
		return false;
	}

	// Initialize the Direct3D object.
	result = m_D3D->Initialize(screenWidth, screenHeight, VSYNC_ENABLED, hwnd, FULL_SCREEN, SCREEN_DEPTH, SCREEN_NEAR);
	if(!result)
	{
		MessageBox(hwnd, L"Could not initialize Direct3D.", L"Error", MB_OK);
		return false;
	}

	// Create the camera object.
	m_Camera = new CameraClass;
	if(!m_Camera)
	{
		return false;
	}

	// Initialize a base view matrix with the camera for 2D user interface rendering.
	m_Camera->SetPosition(0.0f, 0.0f, -1.0f);
	m_Camera->Render();
	m_Camera->GetViewMatrix(baseViewMatrix);

	// Set the initial position of the camera.
	m_Camera->SetPosition(0.0f, 0.0f, -10.0f);
	
	// Create the text object.
	m_Text = new TextClass;
	if(!m_Text)
	{
		return false;
	}

	// Initialize the text object.
	result = m_Text->Initialize(m_D3D->GetDevice(), m_D3D->GetDeviceContext(), hwnd, screenWidth, screenHeight, baseViewMatrix);
	if(!result)
	{
		MessageBox(hwnd, L"Could not initialize the text object.", L"Error", MB_OK);
		return false;
	}

	// Create the model object.
	m_Model = new ModelClass;
	if(!m_Model)
	{
		return false;
	}

	// Initialize the model object.
	result = m_Model->Initialize(m_D3D->GetDevice(), NULL);//L"../Engine/data/seafloor.dds");
	if(!result)
	{
		MessageBox(hwnd, L"Could not initialize the model object.", L"Error", MB_OK);
		return false;
	}

	// Create the texture shader object.
	m_TextureShader = new TextureShaderClass;
	if(!m_TextureShader)
	{
		return false;
	}

	// Initialize the texture shader object.
	result = m_TextureShader->Initialize(m_D3D->GetDevice(), hwnd);
	if(!result)
	{
		MessageBox(hwnd, L"Could not initialize the texture shader object.", L"Error", MB_OK);
		return false;
	}

	// Create the input object. The input object will be used to handle reading the keyboard and mouse input from the user.
	m_Input = new InputClass;
	if(!m_Input)
	{
		return false;
	}

	// Initialize the input object.
	result = m_Input->Initialize(hinstance, hwnd, screenWidth, screenHeight);
	if(!result)
	{
		MessageBox(hwnd, L"Could not initialize the input object.", L"Error", MB_OK);
		return false;
	}

	// Create the font shader object.
	m_FontShader = new FontShaderClass;
	if(!m_FontShader)
	{
		return false;
	}

	// Initialize the font shader object.
	result = m_FontShader->Initialize(m_D3D->GetDevice(), hwnd);
	if(!result)
	{
		MessageBox(hwnd, L"Could not initialize the font shader object.", L"Error", MB_OK);
		return false;
	}

	// Create the fps object.
	m_Fps = new FpsClass;
	if(!m_Fps)
	{
		return false;
	}

	// Initialize the fps object.
	m_Fps->Initialize();

	// Retrieve the video card information.
	m_D3D->GetVideoCardInfo(videoCard, videoMemory);

	// Set the video card information in the text object.
	result = m_Text->SetVideoCardInfo(videoCard, videoMemory, m_D3D->GetDeviceContext());
	if(!result)
	{
		MessageBox(hwnd, L"Could not set video card info in the text object.", L"Error", MB_OK);
		return false;
	}	

	TwInit(TW_DIRECT3D11, m_D3D->GetDevice()); // for Direct3D 11
	TwWindowSize(screenWidth, screenHeight);	
	
	// Initialise everything.
	PARAMS.fVortStrength = 0.1f;
	PARAMS.fSigma = 0.00625f;
	PARAMS.fKappa = 0.25f;
	PARAMS.bDiffusion = false;
	PARAMS.color = (float *) calloc(3, 4);
	PARAMS.color[0] = 1.0f;
	PARAMS.color[1] = 1.0f;
	PARAMS.color[2] = 1.0f;
	PARAMS.bVorticity = false;
	PARAMS.bBuoyancy = true;
	PARAMS.bViscosity = false;
	PARAMS.dt = 0.01f;
	PARAMS.iPoissonIterations = 10;
	PARAMS.fDiffFactor = 0.0f;
	PARAMS.fViscFactor = 0.0f;
	PARAMS.enumCurrentAdvect = SemiMacCormack;
	PARAMS.enumIterativeMethod = GaussSeidel;
	
	TwBar *myBar;
	myBar = TwNewBar("Fluid Simulation");

	{
        TwEnumVal advectionEV[2] = {{SemiMacCormack, "MacCormack"}, {SingleSL, "SingleSL"}};
        TwType advectionType = TwDefineEnum("AdvectionType", advectionEV, 2);
		TwAddVarRW(myBar, "Advection", advectionType, &PARAMS.enumCurrentAdvect, "group = 'Fluid Properties'");
    }
	{
        TwEnumVal iterativeEV[2] = {{GaussSeidel, "GaussSeidel"}, {Jacobi, "Jacobi"}};
        TwType iterativeType = TwDefineEnum("IterativeType", iterativeEV, 2);
		TwAddVarRW(myBar, "Iterative Method", iterativeType, &PARAMS.enumIterativeMethod, "group = 'Fluid Properties'");
    }

	TwAddVarRW(myBar, "Time Step", TW_TYPE_FLOAT, &PARAMS.dt, "group = 'Fluid Properties' min=0 max=2 step=0.01");
	TwAddVarRW(myBar, "Poisson Iterations", TW_TYPE_INT32, &PARAMS.iPoissonIterations, "group = 'Fluid Properties'");
	TwAddVarRW(myBar, "Diffusion factor", TW_TYPE_FLOAT, &PARAMS.fDiffFactor, "group = 'Fluid Properties' min=0 max=1 step=0.0001");
	TwAddVarRW(myBar, "Viscosity factor", TW_TYPE_FLOAT, &PARAMS.fViscFactor, "group = 'Fluid Properties' min=0 max=1 step=0.0001");

	TwAddVarRW(myBar, "Vorticity Strength", TW_TYPE_FLOAT, &PARAMS.fVortStrength, "group = 'Physical Constants' step=0.01");
	TwAddVarRW(myBar, "Kappa", TW_TYPE_FLOAT, &PARAMS.fKappa, "group = 'Physical Constants' step=0.1");
	TwAddVarRW(myBar, "Sigma", TW_TYPE_FLOAT, &PARAMS.fSigma, "group = 'Physical Constants' step=0.1");
	
	TwAddVarRW(myBar, "bgColor", TW_TYPE_COLOR3F, PARAMS.color, " label='Smoke color' ");

	TwAddVarRW(myBar, "Viscosity", TW_TYPE_BOOLCPP, &PARAMS.bViscosity, "group = 'Toggles'");
	TwAddVarRW(myBar, "Buoyancy", TW_TYPE_BOOLCPP, &PARAMS.bBuoyancy, "group = 'Toggles'");
	TwAddVarRW(myBar, "Vorticity Confinement", TW_TYPE_BOOLCPP, &PARAMS.bVorticity, "group = 'Toggles'");
	TwAddVarRW(myBar, "Diffusion", TW_TYPE_BOOLCPP, &PARAMS.bDiffusion, "group = 'Toggles'");

	InitCUDA();

	// Here begins texture initializations.
    g_texture_2d.width  = 256;
    g_texture_2d.height = 256;

    D3D11_TEXTURE2D_DESC desc;
    ZeroMemory(&desc, sizeof(D3D11_TEXTURE2D_DESC));
    desc.Width = g_texture_2d.width;
    desc.Height = g_texture_2d.height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R32_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    if (FAILED(m_D3D->GetDevice()->CreateTexture2D(&desc, NULL, &g_texture_2d.pTexture)))
    {
        return false;
    }

    if (FAILED(m_D3D->GetDevice()->CreateShaderResourceView(g_texture_2d.pTexture, NULL, &g_texture_2d.pSRView)))
    {
        return false;
    }

	 // register the Direct3D resources that we'll use
    // we'll read to and write from g_texture_2d, so don't set any special map flags for it
	cudaGraphicsD3D11RegisterResource(&g_texture_2d.cudaResource, g_texture_2d.pTexture, cudaGraphicsRegisterFlagsNone);
    //getLastCudaError("cudaGraphicsD3D11RegisterResource (g_texture_2d) failed");
    // cuda cannot write into the texture directly : the texture is seen as a cudaArray and can only be mapped as a texture
    // Create a buffer so that cuda can write into it
    // pixel fmt is DXGI_FORMAT_R32G32B32A32_FLOAT
    if (cudaMallocPitch(&g_texture_2d.cudaLinearMemory, &g_texture_2d.pitch, g_texture_2d.width * sizeof(float) * 1, g_texture_2d.height) != cudaSuccess)
    {
		return false;
	}
    cudaMemset(g_texture_2d.cudaLinearMemory, 1, g_texture_2d.pitch * g_texture_2d.height);

	return true;
}


void GraphicsClass::Shutdown()
{
	TwTerminate();
	cudaDeviceSynchronize();

	cudaGraphicsUnregisterResource(g_texture_2d.cudaResource);
    cudaFree(g_texture_2d.cudaLinearMemory);

	g_texture_2d.pSRView->Release();
    g_texture_2d.pTexture->Release();

	dynlinkUnloadD3D11API();

	// Release the fps object.
	if(m_Fps)
	{
		delete m_Fps;
		m_Fps = 0;
	}

	// Release the font shader object.
	if(m_FontShader)
	{
		m_FontShader->Shutdown();
		delete m_FontShader;
		m_FontShader = 0;
	}

	// Release the text object.
	if(m_Text)
	{
		m_Text->Shutdown();
		delete m_Text;
		m_Text = 0;
	}

	// Release the texture shader object.
	if(m_TextureShader)
	{
		m_TextureShader->Shutdown();
		delete m_TextureShader;
		m_TextureShader = 0;
	}

	// Release the model object.
	if(m_Model)
	{
		m_Model->Shutdown();
		delete m_Model;
		m_Model = 0;
	}

	// Release the camera object.
	if(m_Camera)
	{
		delete m_Camera;
		m_Camera = 0;
	}

	// Release the D3D object.
	if(m_D3D)
	{
		m_D3D->Shutdown();
		delete m_D3D;
		m_D3D = 0;
	}

	/// Release the input object.
	if(m_Input)
	{
		m_Input->Shutdown();
		delete m_Input;
		m_Input = 0;
	}

	FreeCUDA();
	delete PARAMS.color;
	PARAMS.color = 0;

	return;
}


bool GraphicsClass::Frame()
{
	bool result;
	int N = 254;

	m_Fps->Frame();

	// Read the user input.
	result = m_Input->Frame();
	if(!result)
	{
		return false;
	}

	// Then handle it. 
	result = HandleInput();
	if(!result)
	{
		return false;
	}	

	
	if (!scenePaused) 
	{
		FrameCUDA(N, source_vel_i, source_vel_j, source_dens_i, source_dens_j, source_u_value, source_v_value, source_dens_value, PARAMS);

		// Map.
        cudaGraphicsResource *ppResource[1] = {g_texture_2d.cudaResource};
        cudaGraphicsMapResources(1, ppResource, 0);
		getLastCudaError("cudaGraphicsMapResources(3) failed");

        cudaArray *cuArray;
        cudaGraphicsSubResourceGetMappedArray(&cuArray, g_texture_2d.cudaResource, 0, 0);
		getLastCudaError("cudaGraphicsSubResourceGetMappedArray (cuda_texture_2d) failed");

		cuda_texture_2d(N, g_texture_2d.cudaLinearMemory, g_texture_2d.width, g_texture_2d.height, g_texture_2d.pitch);
		getLastCudaError("cuda_texture_2d failed");

		cudaMemcpy2DToArray(
            cuArray, // dst array
            0, 0,    // offset
            g_texture_2d.cudaLinearMemory, g_texture_2d.pitch,       // src
            g_texture_2d.width*1*sizeof(float), g_texture_2d.height, // extent
            cudaMemcpyDeviceToDevice); // kind
		getLastCudaError("cudaMemcpy2DToArray failed");

        // Unmap.
        cudaGraphicsUnmapResources(1, ppResource, 0);
		getLastCudaError("cudaGraphicsMapResources(3) failed");

		// Render the graphics scene.
		result = Render();
		if(!result)
		{
			return false;
		}
    }	

	return true;
}

bool GraphicsClass::HandleInput()
{
	bool keyDown, result;
	int N = 254;
	int mouseX, mouseY;	

	source_dens_value = 0.0f;
	source_u_value = 0.0f;
	source_v_value = 0.0f;

	m_Input->GetMouseLocation(mouseX, mouseY);

	if (m_Input->IsMouseRightPressed())
	{
		int i = (int) ((mouseX - 362) / (float) 2.519f);
		int j = (int) ((mouseY - 63) / (float) 2.519f);
		
		if (i<=N && j<=N && i>=1 && j>=1)
		{
			source_dens_value = 50.0f;
			source_dens_i = i;
			source_dens_j = j;
		}
	}

	if (m_Input->IsMouseLeftPressed())
	{
		// get index for fluid cell under mouse position
		int i = (int) ((mouseX - 362) / (float) 2.519f);
		int j = (int) ((mouseY - 63) / (float) 2.519f);
		
		if (i<=N && j<=N && i>=1 && j>=1)
		{
			source_vel_i = i;
			source_vel_j = j;
			source_u_value = 300 * (mouseX - omx);
			source_v_value = 300 * (mouseY - omy);
		}
	}

	m_Input->GetMouseLocation(mouseX, mouseY);
	omx = mouseX;
	omy = mouseY;
	
	// Update the CPU usage value in the text object.
	result = m_Text->SetFps(m_Fps->GetFps(), m_D3D->GetDeviceContext());
	if(!result)
	{
		return false;
	}

	// Pause.
	if (m_Input->IsPJustPressed()) scenePaused = !scenePaused;

	// Check if the user pressed escape and wants to exit the application.
	if(m_Input->IsEscapePressed() == true)
	{
		return false;
	}

	return true;
}

bool GraphicsClass::Render()
{
	D3DXMATRIX worldMatrix, viewMatrix, projectionMatrix, orthoMatrix;
	bool result;

	// Clear the buffers to begin the scene.
	m_D3D->BeginScene(0.0f, 0.0f, 0.0f, 1.0f);

	// Generate the view matrix based on the camera's position.
	m_Camera->Render();

	// Get the world, view, and projection matrices from the camera and d3d objects.
	m_Camera->GetViewMatrix(viewMatrix);
	m_D3D->GetWorldMatrix(worldMatrix);
	m_D3D->GetProjectionMatrix(projectionMatrix);
	m_D3D->GetOrthoMatrix(orthoMatrix);

	// Put the model vertex and index buffers on the graphics pipeline to prepare them for drawing.
	m_Model->Render(m_D3D->GetDeviceContext());

	// Render the model using the texture shader.
	result = m_TextureShader->Render(m_D3D->GetDeviceContext(), m_Model->GetIndexCount(), worldMatrix, viewMatrix, projectionMatrix, g_texture_2d.pSRView, PARAMS.color);
	if(!result)
	{
		return false;
	}

	// Turn off the Z buffer to begin all 2D rendering.
	m_D3D->TurnZBufferOff();

	TwDraw();

	// Render the text strings.
	result = m_Text->Render(m_D3D->GetDeviceContext(), m_FontShader, worldMatrix, orthoMatrix);
	if(!result)
	{
		return false;
	}

	// Turn the Z buffer back on now that all 2D rendering has completed.
	m_D3D->TurnZBufferOn();

	// Present the rendered scene to the screen.
	m_D3D->EndScene();

	return true;
}