// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_CPU_SKELETON_H_INCLUDED__
#define __NBL_ASSET_I_CPU_SKELETON_H_INCLUDED__

#include "nbl/asset/ISkeleton.h"
#include "nbl/asset/ICPUBuffer.h"

namespace nbl
{
namespace asset
{

class ICPUSkeleton : public ISkeleton<ICPUBuffer>, /*TODO: public BlobSerializable, */public IAsset
{
	public:
		using Base = ISkeleton<ICPUBuffer>;

		template<typename NameIterator>
		ICPUSkeleton(SBufferBinding<BufferType>&& _parentJointIDsBinding, SBufferBinding<BufferType>&& _inverseBindPosesBinding, NameIterator begin, NameIterator end) : Base(std::move(_parentJointIDsBinding),std::move(_inverseBindPosesBinding))
		{
			Base::setJointNames<NameIterator>(begin,end);
		}

		//
		inline const core::matrix3x4SIMD& getInversePoseBindMatrix(Base::joint_id_t jointID) const
		{
			return reinterpret_cast<const core::matrix3x4SIMD*>(m_inverseBindPoses.buffer->getPointer()+m_inverseBindPoses.offset)[jointID];
		}
		inline core::matrix3x4SIMD& getInversePoseBindMatrix(Base::joint_id_t jointID)
		{
			return const_cast<core::matrix3x4SIMD&>(const_cast<const ICPUSkeleton*>(this)->getInversePoseBindMatrix(jointID));
		}

		//
		inline const Base::joint_id_t& getParentJointID(Base::joint_id_t jointID) const
		{
			return reinterpret_cast<const Base::joint_id_t*>(m_parentJointIDs.buffer->getPointer()+m_parentJointIDs.offset)[jointID];
		}
		inline Base::joint_id_t& getParentJointID(Base::joint_id_t jointID)
		{
			return const_cast<Base::joint_id_t&>(const_cast<const ICPUSkeleton*>(this)->getParentJointID(jointID));
		}

		//! Serializes mesh to blob for *.baw file format.
		/** @param _stackPtr Optional pointer to stack memory to write blob on. If _stackPtr==NULL, sufficient amount of memory will be allocated.
			@param _stackSize Size of stack memory pointed by _stackPtr.
			@returns Pointer to memory on which blob was written.
		* TODO
		virtual void* serializeToBlob(void* _stackPtr = NULL, const size_t& _stackSize = 0) const override
		{
			return CorrespondingBlobTypeFor<ICPUSkeleton>::type::createAndTryOnStack(this, _stackPtr, _stackSize);
		}
		*/

		virtual void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override
		{
            convertToDummyObject_common(referenceLevelsBelowToConvert);

			if (referenceLevelsBelowToConvert)
			for (auto i=0u; i<getMeshBufferCount(); i++)
				getMeshBuffer(i)->convertToDummyObject(referenceLevelsBelowToConvert-1u);
		}

		_NBL_STATIC_INLINE_CONSTEXPR auto AssetType = ET_SKELETON;
		inline E_TYPE getAssetType() const override { return AssetType; }
/* TODO
		virtual size_t conservativeSizeEstimate() const override
		{
			size_t estimate = m_nameToJointID.size()*sizeof(std::pair<const char*,joint_id_t>);
			// do we add other things to the size estimate?
			return estimate;
		}
*/
		bool canBeRestoredFrom(const IAsset* _other) const override
		{
			auto other = static_cast<const ICPUSkeleton*>(_other);
			if (getJointCount()!=other->getJointCount())
				return false;
			if (!getMeshBuffer(i)->canBeRestoredFrom(other->getMeshBuffer(i)))
				return false;

			return true;
		}

	protected:
		void restoreFromDummy_impl(IAsset* _other, uint32_t _levelsBelow) override
		{
			auto* other = static_cast<ICPUSkeleton*>(_other);

			if (_levelsBelow)
			{
				--_levelsBelow;
				//for (uint32_t i = 0u; i < getMeshBufferCount(); i++)
					//restoreFromDummy_impl_call(getMeshBuffer(i), other->getMeshBuffer(i), _levelsBelow);
			}
		}
};

}
}

#endif
