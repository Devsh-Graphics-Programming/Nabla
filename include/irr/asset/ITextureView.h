#ifndef __IRR_I_TEXTURE_VIEW_H_INCLUDED__
#define __IRR_I_TEXTURE_VIEW_H_INCLUDED__

#include "irr/asset/IDescriptor.h"

namespace irr {
namespace asset
{

class ITextureView : public IDescriptor
{
public:
    E_CATEGORY getTypeCategory() const override { return EC_IMAGE; }

protected:
    ITextureView() = default;
    virtual ~ITextureView() = default;
};

}}

#endif