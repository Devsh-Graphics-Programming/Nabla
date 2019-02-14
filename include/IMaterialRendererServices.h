// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_MATERIAL_RENDERER_SERVICES_H_INCLUDED__
#define __I_MATERIAL_RENDERER_SERVICES_H_INCLUDED__

#include "SMaterial.h"

namespace irr
{
namespace video
{

class IVideoDriver;


enum E_SHADER_CONSTANT_TYPE
{
    ESCT_FLOAT=0,
    ESCT_FLOAT_VEC2,
    ESCT_FLOAT_VEC3,
    ESCT_FLOAT_VEC4,
    ESCT_INT,
    ESCT_INT_VEC2,
    ESCT_INT_VEC3,
    ESCT_INT_VEC4,
    ESCT_UINT,
    ESCT_UINT_VEC2,
    ESCT_UINT_VEC3,
    ESCT_UINT_VEC4,
    ESCT_BOOL,
    ESCT_BOOL_VEC2,
    ESCT_BOOL_VEC3,
    ESCT_BOOL_VEC4,
    ESCT_FLOAT_MAT2,
    ESCT_FLOAT_MAT3,
    ESCT_FLOAT_MAT4,
    ESCT_FLOAT_MAT2x3,
    ESCT_FLOAT_MAT2x4,
    ESCT_FLOAT_MAT3x2,
    ESCT_FLOAT_MAT3x4,
    ESCT_FLOAT_MAT4x2,
    ESCT_FLOAT_MAT4x3,
    ESCT_INVALID_COUNT
};

//! Interface providing some methods for changing advanced, internal states of a IVideoDriver.
class IMaterialRendererServices
{
public:

	//! Can be called by an IMaterialRenderer to make its work easier.
	/** Sets all basic renderstates if needed.
	Basic render states are diffuse, ambient, specular, and emissive color,
	specular power, bilinear and trilinear filtering, wireframe mode,
	grouraudshading, lighting, zbuffer, zwriteenable, backfaceculling and
	fog enabling.
	\param material The new material to be used.
	\param lastMaterial The material used until now.
	\param resetAllRenderstates Set to true if all renderstates should be
	set, regardless of their current state. */
	virtual void setBasicRenderStates(const SGPUMaterial& material,
		const SGPUMaterial& lastMaterial,
		bool resetAllRenderstates) = 0; //depr



	virtual void setShaderConstant(const void* data, int32_t location, E_SHADER_CONSTANT_TYPE type, uint32_t number=1) = 0;


	//! Get pointer to the IVideoDriver interface
	/** \return Pointer to the IVideoDriver interface */
	virtual IVideoDriver* getVideoDriver() = 0; //depr
};

} // end namespace video
} // end namespace irr

#endif

