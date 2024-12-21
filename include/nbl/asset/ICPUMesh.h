// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_CPU_MESH_H_INCLUDED__
#define __NBL_ASSET_I_CPU_MESH_H_INCLUDED__

#include "nbl/asset/IMesh.h"
#include "nbl/asset/IAsset.h"
#include "nbl/asset/ICPUMeshBuffer.h"

namespace nbl
{
namespace asset
{

class ICPUMesh final : public IMesh<ICPUMeshBuffer>, public IAsset
{
	public:
		//! These are not absolute constants, just the most common situation, there may be setups of assets/resources with completely different relationships.
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t MESHBUFFER_HIERARCHYLEVELS_BELOW = 1u;//mesh->meshbuffer
        _NBL_STATIC_INLINE_CONSTEXPR uint32_t PIPELINE_HIERARCHYLEVELS_BELOW = MESHBUFFER_HIERARCHYLEVELS_BELOW+1u;//meshbuffer->pipeline
        _NBL_STATIC_INLINE_CONSTEXPR uint32_t PIPELINE_LAYOUT_HIERARCHYLEVELS_BELOW = PIPELINE_HIERARCHYLEVELS_BELOW +1u;//meshbuffer->pipeline->layout
        _NBL_STATIC_INLINE_CONSTEXPR uint32_t DESC_SET_HIERARCHYLEVELS_BELOW = MESHBUFFER_HIERARCHYLEVELS_BELOW+1u;//meshbuffer->ds
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t IMAGEVIEW_HIERARCHYLEVELS_BELOW = DESC_SET_HIERARCHYLEVELS_BELOW+1u;//ds->imageview
        _NBL_STATIC_INLINE_CONSTEXPR uint32_t IMAGE_HIERARCHYLEVELS_BELOW = IMAGEVIEW_HIERARCHYLEVELS_BELOW+1u;//imageview->image
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t SAMPLER_HIERARCHYLEVELS_BELOW = DESC_SET_HIERARCHYLEVELS_BELOW+2u;//ds->layout->immutable
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t BUFFER_HIERARCHYLEVELS_BELOW = DESC_SET_HIERARCHYLEVELS_BELOW+1u;//ds->buffer
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t VTX_IDX_BUFFER_HIERARCHYLEVELS_BELOW = MESHBUFFER_HIERARCHYLEVELS_BELOW+1u;//meshbuffer->m_vertexBufferBindings or m_indexBufferBinding

		//!
		inline core::SRange<const ICPUMeshBuffer* const> getMeshBuffers() const override
		{
			auto begin = reinterpret_cast<const ICPUMeshBuffer* const*>(m_meshBuffers.data());
			return core::SRange<const ICPUMeshBuffer* const>(begin,begin+m_meshBuffers.size());
		}
		inline core::SRange<ICPUMeshBuffer* const> getMeshBuffers() override
		{
			assert(isMutable());
			auto begin = reinterpret_cast<ICPUMeshBuffer* const*>(m_meshBuffers.data());
			return core::SRange<ICPUMeshBuffer* const>(begin,begin+m_meshBuffers.size());
		}

		//! Mutable access to the vector of meshbuffers
		inline auto& getMeshBufferVector()
		{
			assert(isMutable());
			return m_meshBuffers;
		}

		//!
		inline const core::aabbox3df& getBoundingBox() const // needed so the compiler doesn't freak out
		{
			return IMesh<ICPUMeshBuffer>::getBoundingBox();
		}
		inline void setBoundingBox(const core::aabbox3df& newBoundingBox) override
		{
			assert(isMutable());
			return IMesh<ICPUMeshBuffer>::setBoundingBox(newBoundingBox);
		}


		_NBL_STATIC_INLINE_CONSTEXPR auto AssetType = ET_MESH;
		inline E_TYPE getAssetType() const override { return AssetType; }
		
        core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
        {
            auto cp = core::make_smart_refctd_ptr<ICPUMesh>();
			cp->m_cachedBoundingBox = m_cachedBoundingBox;
            cp->m_meshBuffers.resize(m_meshBuffers.size());

			auto outIt = cp->m_meshBuffers.begin();
			for (auto inIt=m_meshBuffers.begin(); inIt!=m_meshBuffers.end(); inIt++,outIt++)
			{
                if (_depth>0u && *inIt)
					*outIt = core::smart_refctd_ptr_static_cast<ICPUMeshBuffer>((*inIt)->clone(_depth-1u));
				else
	                *outIt = *inIt;
			}

            return cp;
        }

        //! CLASS IS DEPRECATED ANYWAY
		inline size_t getDependantCount() const override {return 0;}

	protected:
		inline IAsset* getDependant_impl(const size_t ix) override {return nullptr;}

	private:
		core::vector<core::smart_refctd_ptr<ICPUMeshBuffer>> m_meshBuffers;
};

}
}

#endif
