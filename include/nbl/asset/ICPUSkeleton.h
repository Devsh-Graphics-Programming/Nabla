// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_I_CPU_SKELETON_H_INCLUDED_
#define _NBL_ASSET_I_CPU_SKELETON_H_INCLUDED_

#include "nbl/asset/ISkeleton.h"
#include "nbl/asset/ICPUBuffer.h"

namespace nbl::asset
{

class ICPUSkeleton final : public ISkeleton<ICPUBuffer>, /*TODO: public BlobSerializable, */public IAsset
{
	public:
		using base_t = ISkeleton<ICPUBuffer>;

		template<typename NameIterator>
		inline ICPUSkeleton(SBufferBinding<ICPUBuffer>&& _parentJointIDsBinding, SBufferBinding<ICPUBuffer>&& _defaultTransforms, NameIterator begin, NameIterator end) :
			base_t(std::move(_parentJointIDsBinding), std::move(_defaultTransforms), std::distance(begin, end))
		{
			if(_parentJointIDsBinding.buffer)
				_parentJointIDsBinding.buffer->addUsageFlags(IBuffer::EUF_STORAGE_BUFFER_BIT);
			if(_defaultTransforms.buffer)
				_defaultTransforms.buffer->addUsageFlags(IBuffer::EUF_STORAGE_BUFFER_BIT);
			base_t::setJointNames<NameIterator>(begin,end);
		}
		template<typename... Args>
		inline ICPUSkeleton(Args&&... args) : base_t(std::forward<Args>(args)...) {}

		//!
		inline const SBufferBinding<ICPUBuffer>& getParentJointIDBinding() const
		{
			return m_parentJointIDs;
		}

		//!
		inline const SBufferBinding<ICPUBuffer>& getDefaultTransformBinding() const
		{
			return m_defaultTransforms;
		}

		//!
		inline const core::matrix3x4SIMD& getDefaultTransformMatrix(base_t::joint_id_t jointID) const
		{
			const uint8_t* ptr = reinterpret_cast<const uint8_t*>(m_defaultTransforms.buffer->getPointer());
			return reinterpret_cast<const core::matrix3x4SIMD*>(ptr+m_defaultTransforms.offset)[jointID];
		}
		inline core::matrix3x4SIMD& getDefaultTransformMatrix(base_t::joint_id_t jointID)
		{
			assert(!isImmutable_debug());
			return const_cast<core::matrix3x4SIMD&>(const_cast<const ICPUSkeleton*>(this)->getDefaultTransformMatrix(jointID));
		}

		//!
		inline const base_t::joint_id_t& getParentJointID(base_t::joint_id_t jointID) const
		{
			const uint8_t* ptr = reinterpret_cast<const uint8_t*>(m_parentJointIDs.buffer->getPointer());
			return reinterpret_cast<const base_t::joint_id_t*>(ptr+m_parentJointIDs.offset)[jointID];
		}

		//! Serializes skeleton to blob for *.nbl file format.
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
			SBufferBinding<ICPUBuffer> _defaultTransformsBinding = { m_defaultTransforms.offset,_depth>0u&&m_defaultTransforms.buffer ? core::smart_refctd_ptr_static_cast<ICPUBuffer>(m_defaultTransforms.buffer->clone(_depth-1u)):m_defaultTransforms.buffer};

 			auto cp = core::make_smart_refctd_ptr<ICPUSkeleton>(std::move(_parentJointIDsBinding),std::move(_defaultTransformsBinding),m_jointCount);
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
				m_defaultTransforms.buffer->convertToDummyObject(referenceLevelsBelowToConvert-1u);
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
            if (m_defaultTransforms.offset != other->m_defaultTransforms.offset)
                return false;
            if ((!m_defaultTransforms.buffer) != (!other->m_defaultTransforms.buffer))
                return false;
            if (m_defaultTransforms.buffer && !m_defaultTransforms.buffer->canBeRestoredFrom(other->m_defaultTransforms.buffer.get()))
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
                if (m_defaultTransforms.buffer)
                    restoreFromDummy_impl_call(m_defaultTransforms.buffer.get(),other->m_defaultTransforms.buffer.get(),_levelsBelow);
			}
		}

		bool isAnyDependencyDummy_impl(uint32_t _levelsBelow) const override
		{
			--_levelsBelow;
			if (m_parentJointIDs.buffer && m_parentJointIDs.buffer->isAnyDependencyDummy(_levelsBelow))
					return true;

			return m_defaultTransforms.buffer && m_defaultTransforms.buffer->isAnyDependencyDummy(_levelsBelow);
		}
};

}

#endif
