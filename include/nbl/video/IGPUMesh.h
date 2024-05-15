// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_GPU_MESH_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_MESH_H_INCLUDED_

#if 0 // rewrite
#include "nbl/asset/IMesh.h"
#include "nbl/video/IGPUMeshBuffer.h"

namespace nbl
{
namespace video
{

class IGPUMesh final : public asset::IMesh<IGPUMeshBuffer>
{
		using MeshBufferRefContainer = core::smart_refctd_dynamic_array<core::smart_refctd_ptr<IGPUMeshBuffer>>;
		MeshBufferRefContainer m_meshBuffers;

	public:
		IGPUMesh(uint32_t meshBufferCount) : m_meshBuffers(core::make_refctd_dynamic_array<MeshBufferRefContainer>(meshBufferCount))
		{
		}

		template<typename MeshBufferIterator>
		IGPUMesh(MeshBufferIterator begin, MeshBufferIterator end) : IGPUMesh(std::distance(begin,end))
		{
			std::copy(begin,end,m_meshBuffers->begin());
		}

		inline auto getMeshBufferIterator()
		{
			return m_meshBuffers->data();
		}

		inline core::SRange<const IGPUMeshBuffer* const> getMeshBuffers() const override
		{
			auto begin = reinterpret_cast<const IGPUMeshBuffer* const*>(m_meshBuffers->data());
			return core::SRange<const IGPUMeshBuffer* const>(begin,begin+m_meshBuffers->size());
		}
		inline core::SRange<IGPUMeshBuffer* const> getMeshBuffers() override
		{
			auto begin = reinterpret_cast<IGPUMeshBuffer* const*>(m_meshBuffers->data());
			return core::SRange<IGPUMeshBuffer* const>(begin,begin+m_meshBuffers->size());
		}
};

}
}
#endif

#endif