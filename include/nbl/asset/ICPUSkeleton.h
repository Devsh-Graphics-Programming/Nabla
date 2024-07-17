// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_I_CPU_SKELETON_H_INCLUDED_
#define _NBL_ASSET_I_CPU_SKELETON_H_INCLUDED_

#include "nbl/asset/ISkeleton.h"
#include "nbl/asset/ICPUBuffer.h"

namespace nbl::asset
{

class ICPUSkeleton final : public ISkeleton<ICPUBuffer>, public IAsset
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
			assert(isMutable());
			return const_cast<core::matrix3x4SIMD&>(const_cast<const ICPUSkeleton*>(this)->getDefaultTransformMatrix(jointID));
		}

		//!
		inline const base_t::joint_id_t& getParentJointID(base_t::joint_id_t jointID) const
		{
			const uint8_t* ptr = reinterpret_cast<const uint8_t*>(m_parentJointIDs.buffer->getPointer());
			return reinterpret_cast<const base_t::joint_id_t*>(ptr+m_parentJointIDs.offset)[jointID];
		}

		inline core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
		{
			SBufferBinding<ICPUBuffer> _parentJointIDsBinding = {m_parentJointIDs.offset,_depth>0u&&m_parentJointIDs.buffer ? core::smart_refctd_ptr_static_cast<ICPUBuffer>(m_parentJointIDs.buffer->clone(_depth-1u)):m_parentJointIDs.buffer};
			SBufferBinding<ICPUBuffer> _defaultTransformsBinding = { m_defaultTransforms.offset,_depth>0u&&m_defaultTransforms.buffer ? core::smart_refctd_ptr_static_cast<ICPUBuffer>(m_defaultTransforms.buffer->clone(_depth-1u)):m_defaultTransforms.buffer};

 			auto cp = core::make_smart_refctd_ptr<ICPUSkeleton>(std::move(_parentJointIDsBinding),std::move(_defaultTransformsBinding),m_jointCount);
			assert(!cp->m_stringPool);
			cp->m_stringPoolSize = m_stringPoolSize;
			cp->m_stringPool = _NBL_NEW_ARRAY(char,m_stringPoolSize);
			memcpy(cp->m_stringPool,m_stringPool,m_stringPoolSize);
			for (auto stringToID : m_nameToJointID)
				cp->m_nameToJointID.emplace(stringToID.first-m_stringPool+cp->m_stringPool,stringToID.second);

			return cp;
		}

		constexpr static inline bool HasDependents = true;

		constexpr static inline auto AssetType = ET_SKELETON;
		inline E_TYPE getAssetType() const override { return AssetType; }

		//!
		inline size_t getDependantCount() const override {return 2;}

	protected:
		inline IAsset* getDependant_impl(const size_t ix) override
		{
			return (ix!=0 ? m_defaultTransforms:m_parentJointIDs).buffer.get();
		}
};

}

#endif
