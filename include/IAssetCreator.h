#ifndef __C_IASSET_CREATOR_H_INCLUDED__
#define __C_IASSET_CREATOR_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "irr/macros.h"
#include "IAsset.h"

namespace irr { namespace asset
{

class IRR_FORCE_EBO IAssetCreator : public core::IReferenceCounted
{
public:
    // (Criss) ??? whatever for now. But how would GeometryCreator look as a derivative of this?
    virtual IAsset* createAsset() const = 0;

protected:
    _IRR_INTERFACE_CHILD_DEFAULT(IAssetCreator);
};

}}//irr::asset

#endif //__C_IASSET_CREATOR_H_INCLUDED__