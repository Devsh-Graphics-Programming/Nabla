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
		using instances_container_t = core::refctd_dynamic_array<SInstance>;

		//!
		inline IMeshMetadata() : m_instances(nullptr) {}
		inline IMeshMetadata(core::smart_refctd_ptr<instances_container_t>&& _instances) : m_instances(std::move(_instances)) {}

		//!
		inline core::SRange<const SInstance> getInstances() const
		{
			if (m_instances)
				return {m_instances->begin(),m_instances->end()};
			return {nullptr,nullptr};
		}

	protected:
		virtual ~IMeshMetadata() = default;

		inline IMeshMetadata& operator=(IMeshMetadata&& other)
		{
			std::swap(m_instances,other.m_instances);
			return *this;
		}

		core::smart_refctd_ptr<instances_container_t> m_instances;
};

}
}

#endif
