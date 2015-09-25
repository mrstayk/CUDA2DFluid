#ifndef _FONTCLASS_H_
#define _FONTCLASS_H_

// *** Author: Martin Staykov
// *** May, 2015
// ***
// *** References:
// *** Abertay University
// *** rastertek.com
// *** NVIDIA CUDA Toolkit

// Inlcudes.
#include <d3d11.h>
#include <d3dx10math.h>
#include <fstream>
using namespace std;

// Class includes.
#include "textureclass.h"

// Class name: FontClass
class FontClass
{
private:
	struct FontType
	{
		float left, right;
		int size;
	};

	struct VertexType
	{
		D3DXVECTOR3 position;
	    D3DXVECTOR2 texture;
	};

public:
	FontClass();
	FontClass(const FontClass&);
	~FontClass();

	bool Initialize(ID3D11Device*, char*, WCHAR*);
	void Shutdown();
	void BuildVertexArray(void*, char*, float, float);

	ID3D11ShaderResourceView* GetTexture();
	
private:
	bool LoadFontData(char*);
	void ReleaseFontData();
	bool LoadTexture(ID3D11Device*, WCHAR*);
	void ReleaseTexture();

private:
	FontType* m_Font;
	TextureClass* m_Texture;
};

#endif