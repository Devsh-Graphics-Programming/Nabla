#ifndef __IRR_I_CPU_BUFFER_VIEW_H_INCLUDED__
#define __IRR_I_CPU_BUFFER_VIEW_H_INCLUDED__

#include "irr/asset/IBufferView.h"
#include "irr/asset/ICPUBuffer.h"
#include "irr/asset/IAsset.h"
#include <utility>

namespace irr {
namespace asset
{

class ICPUBufferView : public IBufferView<ICPUBuffer>, public IAsset
{
public:
    ICPUBufferView(core::smart_refctd_ptr<ICPUBuffer> _buffer, E_FORMAT _format, size_t _offset = 0ull, size_t _size = ICPUBufferView::whole_buffer) :
        IBufferView<ICPUBuffer>(std::move(_buffer), _format, _offset, _size)
    {}

protected:
    virtual ~ICPUBufferView() = default;
};

}}

#endif