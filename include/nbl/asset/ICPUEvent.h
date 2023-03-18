#ifndef __NBL_I_CPU_EVENT_H_INCLUDED__
#define __NBL_I_CPU_EVENT_H_INCLUDED__

#include "nbl/asset/IEvent.h"
#include "nbl/asset/IAsset.h"

namespace nbl {
namespace asset
{

class ICPUEvent final : public IEvent, public IAsset
{
public:
    using IEvent::IEvent;

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
        return ET_EVENT;
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
}

#endif
