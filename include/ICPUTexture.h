#ifndef __IRR_I_CPU_TEXTURE_H_INCLUDED__
#define __IRR_I_CPU_TEXTURE_H_INCLUDED__

#include <utility>
#include <algorithm>
#include "irr/core/Types.h"
#include <cassert>
#include "IAsset.h"
#include "CImageData.h"

namespace irr { namespace asset
{

class ICPUTexture : public IAsset
{
public:
    explicit ICPUTexture(const core::vector<video::CImageData*>& _mipmaps) : m_mipmaps{_mipmaps}
    {
        assert(m_mipmaps.size() != 0u);
        for (const auto& mm : m_mipmaps)
            mm->grab();
        sortMipMaps();
        establishFmt();
    }
    explicit ICPUTexture(core::vector<video::CImageData*>&& _mipmaps) : m_mipmaps{std::move(_mipmaps)}
    {
        assert(m_mipmaps.size() != 0u);
        for (const auto& mm : m_mipmaps)
            mm->grab();
        sortMipMaps();
        establishFmt();
    }

    ~ICPUTexture()
    {
        for (const auto& mm : m_mipmaps)
            mm->drop();
    }

    virtual void convertToDummyObject() override {}

    virtual E_TYPE getAssetType() const override { return IAsset::ET_IMAGE; }

    video::CImageData* getMipMap(uint32_t _mipLvl)
    {
        auto it = std::lower_bound(m_mipmaps.begin(), m_mipmaps.end(), [](const video::CImageData* _a, const video::CImageData* _b) { return _a->getSupposedMipLevel() < _b->getSupposedMipLevel(); });
        if (it != m_mipmaps.end() && (*it)->getSupposedMipLevel()==_mipLvl)
            return *it;
    }
    const video::CImageData* getMipMap(uint32_t _mipLvl) const
    {
        return const_cast<ICPUTexture*>(this)->getMipMap(_mipLvl);
    }

    size_t getMipMapCount() const { return m_mipmaps.size(); }

    uint32_t getLowestMip() const { return m_mipmaps.front()->getSupposedMipLevel(); }
    uint32_t getHighestMip() const { return m_mipmaps.back()->getSupposedMipLevel(); }

    video::ECOLOR_FORMAT getColorFormat() const { return m_colorFormat; }

private:
    void sortMipMaps()
    {
        std::sort(std::begin(m_mipmaps), std::end(m_mipmaps), 
            [](const video::CImageData* _a, const video::CImageData* _b) { return _a->getSupposedMipLevel() < _b->getSupposedMipLevel(); }
        );
    }
    void establishFmt()
    {
        video::ECOLOR_FORMAT fmt = m_mipmaps[0]->getColorFormat();
        for (uint32_t i = 1u; i < m_mipmaps.size(); ++i)
        {
            if (fmt != m_mipmaps[i]->getColorFormat())
            {
                fmt = video::ECF_UNKNOWN;
                break;
            }
        }
        m_colorFormat = fmt;
    }

protected:
    video::ECOLOR_FORMAT m_colorFormat;
    core::vector<video::CImageData*> m_mipmaps;
};

}}//irr::asset

#endif//__IRR_I_CPU_TEXTURE_H_INCLUDED__
