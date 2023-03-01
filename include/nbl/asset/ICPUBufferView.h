// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_CPU_BUFFER_VIEW_H_INCLUDED__
#define __NBL_ASSET_I_CPU_BUFFER_VIEW_H_INCLUDED__


#include <utility>
#include "nbl/core/algorithm/utility.h"

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
		ICPUBufferView(core::smart_refctd_ptr<ICPUBuffer> _buffer, E_FORMAT _format, size_t _offset = 0ull, size_t _size = ICPUBufferView::whole_buffer) :
			IBufferView<ICPUBuffer>(std::move(_buffer), _format, _offset, _size)
		{}

		size_t conservativeSizeEstimate() const override { return sizeof(IBufferView<ICPUBuffer>); }

        core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
        {
            auto buf = (_depth > 0u && m_buffer) ? core::smart_refctd_ptr_static_cast<ICPUBuffer>(m_buffer->clone(_depth-1u)) : m_buffer;
            auto cp = core::make_smart_refctd_ptr<ICPUBufferView>(std::move(buf), m_format, m_offset, m_size);
            clone_common(cp.get());

            return cp;
        }

		void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override
		{
            convertToDummyObject_common(referenceLevelsBelowToConvert);

			if (referenceLevelsBelowToConvert)
				m_buffer->convertToDummyObject(referenceLevelsBelowToConvert-1u);
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

		bool equals(const IAsset* _other) const override
		{
			auto* other = static_cast<const ICPUBufferView*>(_other);
			return compareMembers(other) && m_buffer->equals(other->m_buffer.get());
		}


		bool canBeRestoredFrom(const IAsset* _other) const override
		{
			auto* other = static_cast<const ICPUBufferView*>(_other);
			return compareMembers(other) && m_buffer->canBeRestoredFrom(other->m_buffer.get());
		}

		size_t hash() const override
		{
			size_t seed = 0;
			core::hash_combine(seed, m_buffer->hash());
			core::hash_combine(seed, m_format);
			core::hash_combine(seed, m_offset);
			core::hash_combine(seed, m_size);
			return seed;
		}

	protected:
		void restoreFromDummy_impl(IAsset* _other, uint32_t _levelsBelow) override
		{
			auto* other = static_cast<ICPUBufferView*>(_other);

			if (_levelsBelow)
			{
				restoreFromDummy_impl_call(m_buffer.get(), other->m_buffer.get(), _levelsBelow-1u);
			}
		}

		bool isAnyDependencyDummy_impl(uint32_t _levelsBelow) const override
		{
			return m_buffer->isAnyDependencyDummy(_levelsBelow-1u);
		}

		virtual ~ICPUBufferView() = default;

	private:

		bool compareMembers(const ICPUBufferView* other) const {
			return (m_size == other->m_size) && (m_offset == other->m_offset) && (m_format == other->m_format);
		}
};

}
}

#endif