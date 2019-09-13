#ifndef __IRR_I_GPU_BUFFER_VIEW_H_INCLUDED__
#define __IRR_I_GPU_BUFFER_VIEW_H_INCLUDED__

#include "irr/asset/IBufferView.h"
#include "IGPUBuffer.h"
#include "IRenderableVirtualTexture.h"
#include <utility>

namespace irr {
namespace video
{

class IGPUBufferView : public asset::IBufferView<IGPUBuffer>, public IRenderableVirtualTexture
{
public:
    IGPUBufferView(core::smart_refctd_ptr<IGPUBuffer> _buffer, asset::E_FORMAT _format, size_t _offset = 0ull, size_t _size = IGPUBufferView::whole_buffer) :
        asset::IBufferView<IGPUBuffer>(std::move(_buffer), _format, _offset, _size)
    {}

    E_DIMENSION_COUNT getDimensionality() const override { return EDC_ONE; }

    E_VIRTUAL_TEXTURE_TYPE getVirtualTextureType() const override { return EVTT_BUFFER_OBJECT; }

    asset::E_FORMAT getColorFormat() const override { return getFormat(); }

    //not sure what to do about this????
    core::dimension2du getRenderableSize() const override { return { 0,0 }; }

protected:
    virtual ~IGPUBufferView() = default;
};

}}

#endif