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

    _NBL_STATIC_INLINE_CONSTEXPR auto AssetType = ET_EVENT;
    E_TYPE getAssetType() const override
    {
        return AssetType;
    }
     
    bool equals(const IAsset* _other) const override
	{
        return false; // TODO
	}

	size_t hash(std::unordered_map<IAsset*, size_t>* temporary_hash_cache = nullptr) const override
	{
		size_t seed = AssetType;
        //core::hash_combine(seed, hashMatchInCache(m_shader.get(), temporary_hash_cache));
		return seed; // TODO
	}

private:
    bool compatible(const IAsset* _other) const override {
        return IAsset::compatible(_other);
	}

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
