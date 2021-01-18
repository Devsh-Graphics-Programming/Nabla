// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_CPU_MESH_H_INCLUDED__
#define __NBL_ASSET_I_CPU_MESH_H_INCLUDED__

#include "nbl/asset/IMesh.h"
#include "nbl/asset/IAsset.h"
#include "nbl/asset/ICPUMeshBuffer.h"
#include "nbl/asset/bawformat/blobs/MeshBlob.h"

namespace nbl
{
namespace asset
{

class ICPUMesh : public IMesh<ICPUMeshBuffer>, public BlobSerializable, public IAsset
{
	public:
		//! These are not absolute constants, just the most common situation, there may be setups of assets/resources with completely different relationships.
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t MESHBUFFER_HIERARCHYLEVELS_BELOW = 1u;//mesh->meshbuffer
        _NBL_STATIC_INLINE_CONSTEXPR uint32_t PIPELINE_HIERARCHYLEVELS_BELOW = MESHBUFFER_HIERARCHYLEVELS_BELOW+1u;//meshbuffer->pipeline
        _NBL_STATIC_INLINE_CONSTEXPR uint32_t DESC_SET_HIERARCHYLEVELS_BELOW = MESHBUFFER_HIERARCHYLEVELS_BELOW+1u;//meshbuffer->ds
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t IMAGEVIEW_HIERARCHYLEVELS_BELOW = DESC_SET_HIERARCHYLEVELS_BELOW+1u;//ds->imageview
        _NBL_STATIC_INLINE_CONSTEXPR uint32_t IMAGE_HIERARCHYLEVELS_BELOW = IMAGEVIEW_HIERARCHYLEVELS_BELOW+1u;//imageview->image

		//! recalculates the bounding box
		virtual void recalculateBoundingBox(const bool recomputeSubBoxes = false)
		{
			assert(!isImmutable_debug());

			core::aabbox3df bbox(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);

			const auto count = getMeshBufferCount();
			if (count)
			{
				for (uint32_t i=0u; i<count; i++)
				{
					auto* mb = getMeshBuffer(i);
					if (!mb)
						continue;

					if (recomputeSubBoxes)
						mb->recalculateBoundingBox();

					bbox.addInternalBox(mb->getBoundingBox());
				}
			}
			
			setBoundingBox(std::move(bbox));
		}

		void setBoundingBox(const core::aabbox3df& box) override 
		{ 
			assert(!isImmutable_debug());

			IMesh<ICPUMeshBuffer>::setBoundingBox(box); 
		}

		//
		virtual E_MESH_TYPE getMeshType() const override { return EMT_NOT_ANIMATED; }

		//! Serializes mesh to blob for *.baw file format.
		/** @param _stackPtr Optional pointer to stack memory to write blob on. If _stackPtr==NULL, sufficient amount of memory will be allocated.
			@param _stackSize Size of stack memory pointed by _stackPtr.
			@returns Pointer to memory on which blob was written.
		*/
		virtual void* serializeToBlob(void* _stackPtr = NULL, const size_t& _stackSize = 0) const override
		{
			return CorrespondingBlobTypeFor<ICPUMesh>::type::createAndTryOnStack(this, _stackPtr, _stackSize);
		}

		virtual void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override
		{
            convertToDummyObject_common(referenceLevelsBelowToConvert);

			if (referenceLevelsBelowToConvert)
			for (auto i=0u; i<getMeshBufferCount(); i++)
				getMeshBuffer(i)->convertToDummyObject(referenceLevelsBelowToConvert-1u);
		}

		_NBL_STATIC_INLINE_CONSTEXPR auto AssetType = ET_MESH;
		inline E_TYPE getAssetType() const override { return AssetType; }

		virtual size_t conservativeSizeEstimate() const override { return getMeshBufferCount()*sizeof(void*); }

		bool canBeRestoredFrom(const IAsset* _other) const override
		{
			auto other = static_cast<const ICPUMesh*>(_other);
			if (getMeshBufferCount() == other->getMeshBufferCount())
				return false;
			for (uint32_t i = 0u; i < getMeshBufferCount(); ++i)
				if (!getMeshBuffer(i)->canBeRestoredFrom(other->getMeshBuffer(i)))
					return false;

			return true;
		}

	protected:
		void restoreFromDummy_impl(IAsset* _other, uint32_t _levelsBelow) override
		{
			auto* other = static_cast<ICPUMesh*>(_other);

			if (_levelsBelow)
			{
				--_levelsBelow;
				for (uint32_t i = 0u; i < getMeshBufferCount(); i++)
					restoreFromDummy_impl_call(getMeshBuffer(i), other->getMeshBuffer(i), _levelsBelow);
			}
		}
};

}
}

#endif
