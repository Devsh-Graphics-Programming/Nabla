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

class ICPUSkeleton final : public ISkeleton<ICPUBuffer>, /*TODO: public BlobSerializable, */public IAsset
{
	public:
		using base_t = ISkeleton<ICPUBuffer>;

		template<typename NameIterator>
		inline ICPUSkeleton(SBufferBinding<ICPUBuffer>&& _parentJointIDsBinding, SBufferBinding<ICPUBuffer>&& _inverseBindPosesBinding, NameIterator begin, NameIterator end) : base_t(std::move(_parentJointIDsBinding),std::move(_inverseBindPosesBinding),std::distance(begin,end))
		{
			base_t::setJointNames<NameIterator>(begin,end);
		}
		template<typename... Args>
		inline ICPUSkeleton(Args&&... args) : base_t(std::forward<Args>(args)...) {}

		//
		inline const core::matrix3x4SIMD& getInversePoseBindMatrix(base_t::joint_id_t jointID) const
		{
			const uint8_t* ptr = reinterpret_cast<const uint8_t*>(m_inverseBindPoses.buffer->getPointer());
			return reinterpret_cast<const core::matrix3x4SIMD*>(ptr+m_inverseBindPoses.offset)[jointID];
		}
		inline core::matrix3x4SIMD& getInversePoseBindMatrix(base_t::joint_id_t jointID)
		{
			return const_cast<core::matrix3x4SIMD&>(const_cast<const ICPUSkeleton*>(this)->getInversePoseBindMatrix(jointID));
		}

		//
		inline const base_t::joint_id_t& getParentJointID(base_t::joint_id_t jointID) const
		{
			const uint8_t* ptr = reinterpret_cast<const uint8_t*>(m_parentJointIDs.buffer->getPointer());
			return reinterpret_cast<const base_t::joint_id_t*>(ptr+m_parentJointIDs.offset)[jointID];
		}
		inline base_t::joint_id_t& getParentJointID(base_t::joint_id_t jointID)
		{
			return const_cast<base_t::joint_id_t&>(const_cast<const ICPUSkeleton*>(this)->getParentJointID(jointID));
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

		core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
		{
			SBufferBinding<ICPUBuffer> _parentJointIDsBinding = {m_parentJointIDs.offset,_depth>0u&&m_parentJointIDs.buffer ? core::smart_refctd_ptr_static_cast<ICPUBuffer>(m_parentJointIDs.buffer->clone(_depth-1u)):m_parentJointIDs.buffer};
			SBufferBinding<ICPUBuffer> _inverseBindPosesBinding = {m_inverseBindPoses.offset,_depth>0u&&m_inverseBindPoses.buffer ? core::smart_refctd_ptr_static_cast<ICPUBuffer>(m_inverseBindPoses.buffer->clone(_depth-1u)):m_inverseBindPoses.buffer};

 			auto cp = core::make_smart_refctd_ptr<ICPUSkeleton>(std::move(_parentJointIDsBinding),std::move(_inverseBindPosesBinding),m_jointCount);
			clone_common(cp.get());
			assert(!cp->m_stringPool);
			cp->m_stringPoolSize = m_stringPoolSize;
			cp->m_stringPool = _NBL_NEW_ARRAY(char,m_stringPoolSize);
			memcpy(cp->m_stringPool,m_stringPool,m_stringPoolSize);
			for (auto stringToID : m_nameToJointID)
				cp->m_nameToJointID.emplace(stringToID.first-m_stringPool+cp->m_stringPool,stringToID.second);

			return cp;
		}

		virtual void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override
		{
            convertToDummyObject_common(referenceLevelsBelowToConvert);

			if (referenceLevelsBelowToConvert)
			{
				m_parentJointIDs.buffer->convertToDummyObject(referenceLevelsBelowToConvert-1u);
				m_inverseBindPoses.buffer->convertToDummyObject(referenceLevelsBelowToConvert-1u);
				// TODO: do we clear out the string pool and the name to bone ID mapping?
			}
		}

		_NBL_STATIC_INLINE_CONSTEXPR auto AssetType = ET_SKELETON;
		inline E_TYPE getAssetType() const override { return AssetType; }

		virtual size_t conservativeSizeEstimate() const override
		{
			size_t estimate = sizeof(SBufferBinding<ICPUBuffer>)*2ull;
			estimate += sizeof(uint16_t);
			estimate += m_stringPoolSize;
			estimate += m_nameToJointID.size()*sizeof(std::pair<uint32_t,joint_id_t>);
			// do we add other things to the size estimate?
			return estimate;
		}

		bool canBeRestoredFrom(const IAsset* _other) const override
		{
			auto other = static_cast<const ICPUSkeleton*>(_other);
			// if we decide to get rid of the string pool when converting to dummy, then we need to start checking stringpool and map properties here
            if (m_parentJointIDs.offset != other->m_parentJointIDs.offset)
                return false;
            if ((!m_parentJointIDs.buffer) != (!other->m_parentJointIDs.buffer))
                return false;
            if (m_parentJointIDs.buffer && !m_parentJointIDs.buffer->canBeRestoredFrom(other->m_parentJointIDs.buffer.get()))
                return false;
            if (m_inverseBindPoses.offset != other->m_inverseBindPoses.offset)
                return false;
            if ((!m_inverseBindPoses.buffer) != (!other->m_inverseBindPoses.buffer))
                return false;
            if (m_inverseBindPoses.buffer && !m_inverseBindPoses.buffer->canBeRestoredFrom(other->m_inverseBindPoses.buffer.get()))
                return false;
			if (m_jointCount != other->m_jointCount)
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
				
                if (m_parentJointIDs.buffer)
                    restoreFromDummy_impl_call(m_parentJointIDs.buffer.get(),other->m_parentJointIDs.buffer.get(),_levelsBelow);
                if (m_inverseBindPoses.buffer)
                    restoreFromDummy_impl_call(m_inverseBindPoses.buffer.get(),other->m_inverseBindPoses.buffer.get(),_levelsBelow);
			}
		}

		bool isAnyDependencyDummy_impl(uint32_t _levelsBelow) const override
		{
			--_levelsBelow;
			if (m_parentJointIDs.buffer && m_parentJointIDs.buffer->isAnyDependencyDummy(_levelsBelow))
					return true;

			return m_inverseBindPoses.buffer && m_inverseBindPoses.buffer->isAnyDependencyDummy(_levelsBelow);
		}
};

}
}

#endif
