// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_SHADER_CONSTANT_SET_CALLBACT_H_INCLUDED__
#define __I_SHADER_CONSTANT_SET_CALLBACT_H_INCLUDED__

#include "IReferenceCounted.h"
#include "IMaterialRendererServices.h"

namespace irr
{
namespace video
{
	class IMaterialRendererServices;
	class SMaterial;

struct SConstantLocationNamePair
{
	SConstantLocationNamePair()
	{
		location = -1;
		length = 0;
		type = ESCT_INVALID_COUNT;
	}
    int32_t location;
    int32_t length;
    E_SHADER_CONSTANT_TYPE type;
    core::stringc name;
};

//! Interface making it possible to set constants for gpu programs every frame.
/** Implement this interface in an own class and pass a pointer to it to one of
the methods in IGPUProgrammingServices when creating a shader. The
OnSetConstants method will be called every frame now. */
class IShaderConstantSetCallBack : public virtual IReferenceCounted
{
public:
    virtual void PreLink(uint32_t program) {}
    virtual void PostLink(video::IMaterialRendererServices* services, const E_MATERIAL_TYPE &materialType, const irr::core::array<SConstantLocationNamePair> &constants) =0;

	//! Called to let the callBack know the used material (optional method)
	/**
	 \code
	class MyCallBack : public IShaderConstantSetCallBack
	{
		const video::SMaterial *UsedMaterial;

		OnSetMaterial(const video::SMaterial& material)
		{
			UsedMaterial=&material;
		}

		OnSetConstants(IMaterialRendererServices* services, int32_t userData)
		{
			services->setVertexShaderConstant("myColor", reinterpret_cast<float*>(&UsedMaterial->color), 4);
		}
	}
	\endcode
	*/
	virtual void OnSetMaterial(IMaterialRendererServices* services, const SMaterial& material,const SMaterial& lastMaterial) { }

	//! Called by the engine when the vertex and/or pixel shader constants for an material renderer should be set.
	/**
	Implement the IShaderConstantSetCallBack in an own class and implement your own
	OnSetConstants method using the given IMaterialRendererServices interface.
	Pass a pointer to this class to one of the methods in IGPUProgrammingServices
	when creating a shader. The OnSetConstants method will now be called every time
	before geometry is being drawn using your shader material. A sample implementation
	would look like this:
	\code
	virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
	{
		video::IVideoDriver* driver = services->getVideoDriver();

		// set clip matrix at register 4
		core::matrix4 worldViewProj(driver->getTransform(video::ETS_PROJECTION));
		worldViewProj *= driver->getTransform(video::E4X3TS_VIEW);
		worldViewProj *= driver->getTransform(video::E4X3TS_WORLD);
		services->setVertexShaderConstant(&worldViewProj.M[0], 4, 4);
		// for high level shading languages, this would be another solution:
		//services->setVertexShaderConstant("mWorldViewProj", worldViewProj.M, 16);

		// set some light color at register 9
		video::SColorf col(0.0f,1.0f,1.0f,0.0f);
		services->setVertexShaderConstant(reinterpret_cast<const float*>(&col), 9, 1);
		// for high level shading languages, this would be another solution:
		//services->setVertexShaderConstant("myColor", reinterpret_cast<float*>(&col), 4);
	}
	\endcode
	\param services: Pointer to an interface providing methods to set the constants for the shader.
	\param userData: Userdata int which can be specified when creating the shader.
	*/
	virtual void OnSetConstants(IMaterialRendererServices* services, int32_t userData) = 0;

	virtual void OnUnsetMaterial() = 0;
};


} // end namespace video
} // end namespace irr

#endif

