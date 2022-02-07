#ifndef __NBL_I_CPU_RENDERPASS_H_INCLUDED__
#define __NBL_I_CPU_RENDERPASS_H_INCLUDED__

#include "nbl/asset/IAsset.h"
#include "nbl/asset/IRenderpass.h"

namespace nbl
{
namespace asset
{
class ICPURenderpass : public IRenderpass, public IAsset
{
public:
    using IRenderpass::IRenderpass;

    size_t conservativeSizeEstimate() const { return 0ull; /*TODO*/ }
    core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
    {
        // TODO
        return nullptr;
    }
    bool canBeRestoredFrom(const IAsset* _other) const override
    {
        return false;  // TODO
    }
    E_TYPE getAssetType() const override
    {
        return ET_RENDERPASS;
    }

    ~ICPURenderpass() = default;

private:
    void restoreFromDummy_impl(IAsset* _other, uint32_t _levelsBelow) override
    {
        // TODO
    }
    void convertToDummyObject(uint32_t referenceLevelsBelowToConvert = 0u) override
    {
        // TODO
    }
};

}
}

#endif