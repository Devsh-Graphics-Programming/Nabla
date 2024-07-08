#ifndef _NBL_I_CPU_RENDERPASS_H_INCLUDED_
#define _NBL_I_CPU_RENDERPASS_H_INCLUDED_


#include "nbl/asset/IAsset.h"
#include "nbl/asset/IRenderpass.h"


namespace nbl::asset
{

class ICPURenderpass : public IRenderpass, public IAsset
{
    public:
        using IRenderpass::IRenderpass;

        core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
        {
            // TODO
            return nullptr;
        }
        E_TYPE getAssetType() const override
        {
            return ET_RENDERPASS;
        }

        ~ICPURenderpass() = default;
};

}
#endif