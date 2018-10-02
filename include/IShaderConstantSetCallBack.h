// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_SHADER_CONSTANT_SET_CALLBACT_H_INCLUDED__
#define __I_SHADER_CONSTANT_SET_CALLBACT_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "IMaterialRendererServices.h"
#include "SMaterial.h"

namespace irr
{
namespace video
{
	class IMaterialRendererServices;

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
class IShaderConstantSetCallBack : public virtual core::IReferenceCounted
{
public:
    virtual void PreLink(uint32_t program) {}
    virtual void PostLink(video::IMaterialRendererServices* services, const E_MATERIAL_TYPE &materialType, const core::vector<SConstantLocationNamePair> &constants) =0;

	//! Called to let the callBack know the used material (optional method)
	/**
	 \code
	class MyCallBack : public IShaderConstantSetCallBack
	{
		const video::SGPUMaterial *UsedMaterial;

		OnSetMaterial(const video::SGPUMaterial& material)
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
	virtual void OnSetMaterial(IMaterialRendererServices* services, const SGPUMaterial& material,const SGPUMaterial& lastMaterial) { }

	virtual void OnSetConstants(IMaterialRendererServices* services, int32_t userData) = 0;

	virtual void OnUnsetMaterial() = 0;
};


} // end namespace video
} // end namespace irr

#endif

