#ifndef __IRR_I_CPU_TEXTURE_H_INCLUDED__
#define __IRR_I_CPU_TEXTURE_H_INCLUDED__

#include <utility>
#include <algorithm>
#include <cassert>
#include "irr/core/Types.h"
#include "IAsset.h"
#include "CImageData.h"

namespace irr { namespace asset
{

class ICPUTexture : public IAsset
{
protected:
    video::ECOLOR_FORMAT m_colorFormat;
    core::vector<video::CImageData*> m_mipmaps;

    using IteratorType = typename decltype(m_mipmaps)::iterator;
    using ConstIteratorType = typename decltype(m_mipmaps)::const_iterator;

public:
    using RangeType = std::pair<IteratorType, IteratorType>;
    using ConstRangeType = std::pair<ConstIteratorType, ConstIteratorType>;

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

    //! @returns {end(), end()} if _mipLvl level mipmap is not present.
    inline RangeType getMipMap(uint32_t _mipLvl)
    {
        size_t l = 0u, r = m_mipmaps.size()-1u;
        IteratorType foundItr = m_mipmaps.end();
        while (l <= r)
        {
            const size_t i = l + (r-l)/2u;
            if (m_mipmaps[i]->getSupposedMipLevel() == _mipLvl)
            {
                foundItr = m_mipmaps.begin()+i;
                break;
            }
            if (m_mipmaps[i]->getSupposedMipLevel() < _mipLvl)
                l = i+1u;
            else r = i-1u;
        }
        if (foundItr == m_mipmaps.end())
            return std::make_pair(foundItr, foundItr);
        RangeType rng;
        rng.first = rng.second = foundItr;
        while (rng.second != m_mipmaps.end() && (*rng.second)->getSupposedMipLevel())
            ++rng.second;
        return rng;
    }
    //! @returns {end(), end()} if _mipLvl level mipmap is not present.
    inline ConstRangeType getMipMap(uint32_t _mipLvl) const
    {
        return const_cast<ICPUTexture*>(this)->getMipMap(_mipLvl);
    }

    inline size_t getMipMapCount() const { return m_mipmaps.back()->getSupposedMipLevel(); }

    inline uint32_t getLowestMip() const { return m_mipmaps.front()->getSupposedMipLevel(); }
    inline uint32_t getHighestMip() const { return m_mipmaps.back()->getSupposedMipLevel(); }

    inline video::ECOLOR_FORMAT getColorFormat() const { return m_colorFormat; }

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
};

}}//irr::asset

#endif//__IRR_I_CPU_TEXTURE_H_INCLUDED__
