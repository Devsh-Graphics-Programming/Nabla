#ifndef __IRR_I_TEXTURE_VIEW_H_INCLUDED__
#define __IRR_I_TEXTURE_VIEW_H_INCLUDED__

#include "irr/asset/IDescriptor.h"

namespace irr {
namespace asset
{

class ITextureView : public IDescriptor
{
protected:
    ITextureView() = default;
    virtual ~ITextureView() = default;
};

}}

#endif