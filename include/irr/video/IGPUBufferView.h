#ifndef __IRR_I_GPU_BUFFER_VIEW_H_INCLUDED__
#define __IRR_I_GPU_BUFFER_VIEW_H_INCLUDED__

#include "irr/asset/IBufferView.h"
#include "IGPUBuffer.h"
#include <utility>

namespace irr {
namespace video
{

class IGPUBufferView : public asset::IBufferView<IGPUBuffer>, public core::IReferenceCounted
{
public:
    IGPUBufferView(core::smart_refctd_ptr<IGPUBuffer> _buffer, asset::E_FORMAT _format, size_t _offset = 0ull, size_t _size = IGPUBufferView::whole_buffer) :
        asset::IBufferView<IGPUBuffer>(std::move(_buffer), _format, _offset, _size)
    {}

protected:
    virtual ~IGPUBufferView() = default;
};

}}

#endif