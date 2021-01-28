// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_MESH_METADATA_H_INCLUDED__
#define __NBL_ASSET_I_MESH_METADATA_H_INCLUDED__

#include "nbl/asset/ICPUMesh.h"

namespace nbl
{
namespace asset
{

//! 
class IMeshMetadata : public core::Interface
{
	public:
		struct SInstance
		{
			core::matrix3x4SIMD worldTform;
		};

        //! Returns the exact (or guessed) color semantic of the pixel data stored
		const auto& getInstances() const { return m_instances->data; }

	protected:
		//inline IMeshMetadata(const ColorSemantic& _colorSemantic) : colorSemantic(_colorSemantic) {}
		virtual ~IMeshMetadata() = default;

		core::smart_refctd_dynamic_array<SInstance> m_instances;
};

}
}

#endif
