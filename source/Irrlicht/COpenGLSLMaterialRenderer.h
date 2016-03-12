// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_OPENGL_SHADER_LANGUAGE_MATERIAL_RENDERER_H_INCLUDED__
#define __C_OPENGL_SHADER_LANGUAGE_MATERIAL_RENDERER_H_INCLUDED__

#include "IrrCompileConfig.h"
#ifdef _IRR_COMPILE_WITH_OPENGL_

#include "COpenGLExtensionHandler.h"


#include "IMaterialRenderer.h"
#include "IMaterialRendererServices.h"
#include "IShaderConstantSetCallBack.h"
#include "IGPUProgrammingServices.h"
#include "irrArray.h"
#include "irrString.h"

namespace irr
{
namespace video
{

class COpenGLDriver;
class IShaderConstantSetCallBack;

//! Class for using GLSL shaders with OpenGL
//! Please note: This renderer implements its own IMaterialRendererServices
class COpenGLSLMaterialRenderer : public IMaterialRenderer, public IMaterialRendererServices
{
public:

	//! Constructor
	COpenGLSLMaterialRenderer(
		COpenGLDriver* driver,
		s32& outMaterialTypeNr,
		const c8* vertexShaderProgram = 0,
		const c8* vertexShaderEntryPointName = 0,
		const c8* pixelShaderProgram = 0,
		const c8* pixelShaderEntryPointName = 0,
		const c8* geometryShaderProgram = 0,
		const c8* geometryShaderEntryPointName = "main",
		const c8* controlShaderProgram = 0,
		const c8* controlShaderEntryPointName="main",
		const c8* evaluationShaderProgram = 0,
		const c8* evaluationShaderEntryPointName="main",
		u32 patchVertices=3,
		IShaderConstantSetCallBack* callback = 0,
		E_MATERIAL_TYPE baseMaterial = EMT_SOLID,
		s32 userData = 0);

	//! Destructor
	virtual ~COpenGLSLMaterialRenderer();

	virtual void OnSetMaterial(const SMaterial& material, const SMaterial& lastMaterial,
		bool resetAllRenderstates, IMaterialRendererServices* services);

	virtual bool OnRender(IMaterialRendererServices* service);

	virtual void OnUnsetMaterial();

	//! Returns if the material is transparent.
	virtual bool isTransparent() const;

	//! Returns if it's a tessellation shader.
	virtual bool isTessellation() const {return tessellationPatchVertices!=-1;}

	// implementations for the render services
	virtual void setBasicRenderStates(const SMaterial& material, const SMaterial& lastMaterial, bool resetAllRenderstates);
	virtual void setShaderConstant(const void* data, s32 location, E_SHADER_CONSTANT_TYPE type, u32 number=1);
    virtual void setShaderTextures(const s32* textureIndices, s32 location, E_SHADER_CONSTANT_TYPE type, u32 number=1);
	virtual IVideoDriver* getVideoDriver();

protected:

	//! constructor only for use by derived classes who want to
	//! create a fall back material for example.
	COpenGLSLMaterialRenderer(COpenGLDriver* driver,
					IShaderConstantSetCallBack* callback,
					E_MATERIAL_TYPE baseMaterial,
					s32 userData=0);

	void init(s32& outMaterialTypeNr,
		const c8* vertexShaderProgram,
		const c8* pixelShaderProgram,
		const c8* geometryShaderProgram,
		const c8* controlShaderProgram,
		const c8* evaluationShaderProgram,
		u32 patchVertices=3);

	bool createProgram();
	bool createShader(GLenum shaderType, const char* shader);
	bool linkProgram();

	COpenGLDriver* Driver;
	IShaderConstantSetCallBack* CallBack;
	E_MATERIAL_TYPE BaseMaterial;

	s32 tessellationPatchVertices;//is a Tesselation Shader?

	GLuint Program2;
	GLint activeUniformCount;
	s32 UserData;
#ifdef _DEBUG
    core::array<SConstantLocationNamePair> debugConstants;
    core::array<GLuint> debugConstantIndices;
#endif
};


} // end namespace video
} // end namespace irr

#endif // compile with OpenGL
#endif // if included

