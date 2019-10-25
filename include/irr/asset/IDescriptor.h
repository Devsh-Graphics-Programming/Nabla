#ifndef __IRR_I_DESCRIPTOR_H_INCLUDED__
#define __IRR_I_DESCRIPTOR_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"

namespace irr {
namespace asset
{

class IDescriptor : public virtual core::IReferenceCounted
{
public:
    virtual ~IDescriptor() = default;
};

}}

#endif