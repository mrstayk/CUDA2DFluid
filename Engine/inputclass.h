#ifndef _INPUTCLASS_H_
#define _INPUTCLASS_H_

// *** Author: Martin Staykov
// *** May, 2015
// ***
// *** References:
// *** Abertay University
// *** rastertek.com
// *** NVIDIA CUDA Toolkit

// Pre-processing directives.
#define DIRECTINPUT_VERSION 0x0800

// Linking.
#pragma comment(lib, "dinput8.lib")
#pragma comment(lib, "dxguid.lib")

// Includes.
#include <dinput.h>

class InputClass
{
public:
	InputClass();
	InputClass(const InputClass&);
	~InputClass();

	bool Initialize(HINSTANCE hinstance, HWND hwnd, int screenWidth, int screenHeight);
	bool Frame();
	void Shutdown();

	void GetMouseLocation(int&, int&);
	void GetMouseChange(int&,int&);
	int  GetMouseScroll();
	bool IsMouseRightPressed();
	bool IsMouseLeftPressed();

	void KeyDown(unsigned int);
	void KeyUp(unsigned int);

	bool IsKeyDown(unsigned int);

	bool IsLeftPressed();
	bool IsRightPressed();
	bool IsRPressed();
	bool IsPJustPressed();
	bool IsEscapePressed();

private:
	bool ReadMouse();
	bool ReadKeyboard();
	void ProcessInput();

	bool m_keys[256];

	IDirectInput8* m_directInput;
	IDirectInputDevice8* m_keyboard;
	IDirectInputDevice8* m_mouse;
	DIMOUSESTATE m_mouseState;

	unsigned char m_keyboardState[256];
	int m_screenWidth, m_screenHeight;
	int m_mouseX, m_mouseY;

	bool m_prevPressed_P;
};

#endif