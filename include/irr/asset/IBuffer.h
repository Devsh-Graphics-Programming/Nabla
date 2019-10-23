#ifndef __IRR_ASSET_I_BUFFER_H_INCLUDED__
#define __IRR_ASSET_I_BUFFER_H_INCLUDED__

#include "irr/core/IBuffer.h"
#include "irr/asset/IDescriptor.h"

namespace irr { namespace asset
{

class IBuffer : public core::IBuffer, public IDescriptor
{
protected:
    IBuffer() = default;
    virtual ~IBuffer() = default;
};

}}//irr::asset

#endif//__IRR_ASSET_I_BUFFER_H_INCLUDED__