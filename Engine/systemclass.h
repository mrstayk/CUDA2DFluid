#ifndef _SYSTEMCLASS_H_
#define _SYSTEMCLASS_H_

// *** Author: Martin Staykov
// *** May, 2015
// ***
// *** References:
// *** Abertay University
// *** rastertek.com
// *** NVIDIA CUDA Toolkit

#define WIN32_LEAN_AND_MEAN

#include <windows.h>

#include "graphicsclass.h"

class SystemClass
{
public:
	SystemClass();
	SystemClass(const SystemClass&);
	~SystemClass();

	bool Initialize();
	void Shutdown();
	void Run();

	LRESULT CALLBACK MessageHandler(HWND, UINT, WPARAM, LPARAM);

private:
	bool Frame();
	void InitializeWindows(int&, int&);
	void ShutdownWindows();

private:
	LPCWSTR m_applicationName;
	HINSTANCE m_hinstance;
	HWND m_hwnd;

	GraphicsClass* m_Graphics;
};

static LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

static SystemClass* ApplicationHandle = 0;

#endif