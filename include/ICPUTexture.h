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

    //! Finds range of images making up _mipLvl miplevel or higher if _mipLvl miplevel is not present. 
    //! @returns {end(), end()} if _mipLvl > highest present mip level.
    inline RangeType getMipMap(uint32_t _mipLvl)
    {
        if (_impLvl > getMipMapCount())
            return {m_mipmaps.end(), m_mipmaps.end()};
        IteratorType l = m_mipmaps.begin(), it;
        int32_t cnt = m_mipmaps.size();
        int32_t step;

        while (cnt > 0u)
        {
            it = l;
            step = cnt/2;
            it += step;
            if ((*it)->getSupposedMipLevel() < _mipLvl)
            {
                l = ++it;
                cnt -= step+1;
            }
            else cnt = step;
        }
        if (l == m_mipmaps.end())
            return std::make_pair(l, l);
        RangeType rng;
        rng.first = rng.second = l;
        while (rng.second != m_mipmaps.end() && (*rng.second)->getSupposedMipLevel() == (*rng.first)->getSupposedMipLevel())
            ++rng.second;
        return rng;
    }
    //! @returns {end(), end()} if _mipLvl level mipmap is not present.
    inline ConstRangeType getMipMap(uint32_t _mipLvl) const
    {
        return const_cast<ICPUTexture*>(this)->getMipMap(_mipLvl);
    }

    inline size_t getMipMapCount() const { return m_mipmaps.back()->getSupposedMipLevel(); }

    inline uint32_t getHighestMip() const { return m_mipmaps.back()->getSupposedMipLevel(); }

    inline video::ECOLOR_FORMAT getColorFormat() const { return m_colorFormat; }

private:
    inline void sortMipMaps()
    {
        std::sort(std::begin(m_mipmaps), std::end(m_mipmaps), 
            [](const video::CImageData* _a, const video::CImageData* _b) { return _a->getSupposedMipLevel() < _b->getSupposedMipLevel(); }
        );
    }
    inline void establishFmt()
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
