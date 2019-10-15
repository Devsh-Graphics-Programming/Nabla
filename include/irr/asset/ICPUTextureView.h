#ifndef __IRR_I_CPU_TEXTURE_VIEW_H_INCLUDED__
#define __IRR_I_CPU_TEXTURE_VIEW_H_INCLUDED__

#include "irr/asset/IAsset.h"
#include "irr/asset/ITextureView.h"

namespace irr {
namespace asset
{

class ICPUTextureView : public IAsset, public ITextureView
{
public:
    size_t conservativeSizeEstimate() const override { return 0ull /*TODO*/; }
    void convertToDummyObject() override { } //possibly TODO
    E_TYPE getAssetType() const override { return ET_TEXTURE_VIEW; }

protected:
    virtual ~ICPUTextureView() = default;
};

}}

#endif