#ifndef __IRR_I_CPU_BUFFER_VIEW_H_INCLUDED__
#define __IRR_I_CPU_BUFFER_VIEW_H_INCLUDED__


#include <utility>

#include "irr/asset/IAsset.h"
#include "irr/asset/IBufferView.h"
#include "irr/asset/ICPUBuffer.h"

namespace irr
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
            if (isDummyObjectForCacheAliasing)
                return;
            convertToDummyObject_common(referenceLevelsBelowToConvert);

			if (referenceLevelsBelowToConvert)
				m_buffer->convertToDummyObject(referenceLevelsBelowToConvert-1u);
		}
		E_TYPE getAssetType() const override { return ET_BUFFER_VIEW; }

		ICPUBuffer* getUnderlyingBuffer() { return m_buffer.get(); }
		const ICPUBuffer* getUnderlyingBuffer() const { return m_buffer.get(); }

		inline void setOffsetInBuffer(size_t _offset) { m_offset = _offset; }
		inline void setSize(size_t _size) { m_size = _size; }

	protected:
		virtual ~ICPUBufferView() = default;
};

}
}

#endif