#ifndef _NBL_I_CPU_FRAMEBUFFER_H_INCLUDED_
#define _NBL_I_CPU_FRAMEBUFFER_H_INCLUDED_

#include "nbl/asset/IFramebuffer.h"
#include "nbl/asset/IAsset.h"
#include "nbl/asset/ICPUImageView.h"
#include "nbl/asset/ICPURenderpass.h"

namespace nbl::asset
{

class ICPUFramebuffer final : public IAsset, public IFramebuffer<ICPURenderpass,ICPUImageView>
{
    using base_t = IFramebuffer<ICPURenderpass,ICPUImageView>;

public:
    using base_t::base_t;

    ~ICPUFramebuffer() = default;

    size_t conservativeSizeEstimate() const override { return 0ull; } // TODO

    core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
    {
        return nullptr; // TODO
    }

    bool canBeRestoredFrom(const IAsset* _other) const
    {
        return false; // TODO
    }

    E_TYPE getAssetType() const override
    {
        return ET_FRAMEBUFFER;
    }

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
#endif
