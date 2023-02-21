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
		core::SRange<const SInstance> m_instances;

	protected:
		IMeshMetadata() : m_instances(nullptr,nullptr) {}
		IMeshMetadata(core::SRange<const SInstance>&& _instances) : m_instances(std::move(_instances)) {}
		virtual ~IMeshMetadata() = default;

		inline IMeshMetadata& operator=(IMeshMetadata&& other)
		{
			m_instances = other.m_instances;
			return *this;
		}
};

}
}

#endif
