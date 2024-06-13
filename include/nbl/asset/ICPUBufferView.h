// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_CPU_BUFFER_VIEW_H_INCLUDED__
#define __NBL_ASSET_I_CPU_BUFFER_VIEW_H_INCLUDED__


#include <utility>

#include "nbl/asset/IAsset.h"
#include "nbl/asset/IBufferView.h"
#include "nbl/asset/ICPUBuffer.h"

namespace nbl
{
namespace asset
{

class ICPUBufferView : public IBufferView<ICPUBuffer>, public IAsset
{
	public:
		ICPUBufferView(const SBufferRange<ICPUBuffer>& _underlying, const E_FORMAT _format) : IBufferView<ICPUBuffer>(_underlying, _format) {}

		size_t conservativeSizeEstimate() const override { return sizeof(IBufferView<ICPUBuffer>); }

        core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
        {
            auto buf = (_depth > 0u && m_buffer) ? core::smart_refctd_ptr_static_cast<ICPUBuffer>(m_buffer->clone(_depth-1u)) : m_buffer;
			auto cp = core::make_smart_refctd_ptr<ICPUBufferView>(SBufferRange<ICPUBuffer>{m_offset, m_size, m_buffer}, m_format);
            clone_common(cp.get());

            return cp;
        }

		_NBL_STATIC_INLINE_CONSTEXPR auto AssetType = ET_BUFFER_VIEW;
		inline IAsset::E_TYPE getAssetType() const override { return AssetType; }

		ICPUBuffer* getUnderlyingBuffer() 
		{
			assert(!isImmutable_debug());
			return m_buffer.get(); 
		}
		const ICPUBuffer* getUnderlyingBuffer() const { return m_buffer.get(); }

		inline void setOffsetInBuffer(size_t _offset) 
		{
			assert(!isImmutable_debug());
			m_offset = _offset;
		}
		inline void setSize(size_t _size) 
		{
			assert(!isImmutable_debug());
			m_size = _size;
		}


	protected:

		bool compatible(const IAsset* _other) const override {
			auto* other = static_cast<const ICPUBufferView*>(_other);
			if (m_size != other->m_size)
				return false;
			if (m_offset != other->m_offset)
				return false;
			if (m_format != other->m_format)
				return false;
			return true;

		}

        virtual uint32_t getDependencyCount() const override { return  1; }

        virtual core::smart_refctd_ptr<IAsset> getDependency(uint32_t index) const override {
                return index == 0 ? m_buffer : nullptr;
        }

		void hash_impl(size_t& seed) const override {
			core::hash_combine(seed, m_size);
			core::hash_combine(seed, m_offset);
			core::hash_combine(seed, m_format);
		}

		virtual ~ICPUBufferView() = default;
};

}
}

#endif