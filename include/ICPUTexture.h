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

private:
    template<typename VectorRef>
    inline static ICPUTexture* create_impl(VectorRef&& _mipmaps)
    {
        if (!_mipmaps.size())
            return nullptr;

        bool check[17]{ 0 };
        uint32_t sizes[17][3];
        memset(sizes, 0, sizeof(sizes));

        bool allUnknown = true;
        video::ECOLOR_FORMAT colorFmt = _mipmaps.front()->getColorFormat();
        for (const video::CImageData* img : _mipmaps)
        {
            const uint32_t lvl = img->getSupposedMipLevel();
            check[lvl] = true;
            if (lvl > 16u)
                return nullptr;

            const video::ECOLOR_FORMAT fmt = img->getColorFormat();
            if (fmt != video::ECF_UNKNOWN)
                allUnknown = false;
            if (fmt != colorFmt && fmt != video::ECF_UNKNOWN)
                return nullptr;
            if (colorFmt == video::ECF_UNKNOWN)
                colorFmt = fmt;

            for (uint32_t i = 0u; i < 3u; ++i)
                if (sizes[lvl][i] < img->getSliceMax()[i])
                    sizes[lvl][i] = img->getSliceMax()[i];
        }
        if (allUnknown)
            return nullptr;

        auto tooLarge = [](const uint32_t* _sz, uint32_t _l) -> bool {
            for (uint32_t d = 0u; d < 3u; ++d)
                if (_sz[d] > (0x10000u>>_l))
                    return true;
            return false;
        };
        for (uint32_t i = 0u; i < 17u; ++i)
            if (check[i] && tooLarge(sizes[i], i))
                return nullptr;

        return new ICPUTexture(std::forward<decltype(_mipmaps)>(_mipmaps));
    }

public:
    inline static ICPUTexture* create(const core::vector<video::CImageData*>& _mipmaps)
    {
        return create_impl(_mipmaps);
    }
    inline static ICPUTexture* create(core::vector<video::CImageData*>&& _mipmaps)
    {
        return create_impl(std::move(_mipmaps));
    }
    template<typename Iter>
    inline static ICPUTexture* create(Iter _first, Iter _last)
    {
        return create(core::vector<video::CImageData*>(_first, _last));
    }

protected:
    explicit ICPUTexture(const core::vector<video::CImageData*>& _mipmaps) : m_mipmaps{_mipmaps}, m_colorFormat{video::ECF_UNKNOWN}
    {
        for (const auto& mm : m_mipmaps)
            mm->grab();
        if (m_mipmaps.size())
        {
            sortMipMaps();
            establishFmt();
        }
    }
    explicit ICPUTexture(core::vector<video::CImageData*>&& _mipmaps) : m_mipmaps{std::move(_mipmaps)}, m_colorFormat{video::ECF_UNKNOWN}
    {
        for (const auto& mm : m_mipmaps)
            mm->grab();
        if (m_mipmaps.size())
        {
            sortMipMaps();
            establishFmt();
        }
    }

    ~ICPUTexture()
    {
        for (const auto& mm : m_mipmaps)
            mm->drop();
    }

public:
    virtual void convertToDummyObject() override {}

    virtual E_TYPE getAssetType() const override { return IAsset::ET_IMAGE; }

    virtual size_t conservativeSizeEstimate() const override
    {
        return getCacheKey().length()+1u;
    }

    const core::vector<video::CImageData*>& getMipmaps() const { return m_mipmaps; }

    //! Finds range of images making up _mipLvl miplevel or higher if _mipLvl miplevel is not present. 
    //! @returns {end(), end()} if _mipLvl > highest present mip level.
    inline RangeType getMipMap(uint32_t _mipLvl)
    {
        if (_mipLvl > getHighestMip())
            return {m_mipmaps.end(), m_mipmaps.end()};
        IteratorType l = m_mipmaps.begin(), it;
        int32_t cnt = m_mipmaps.size();
        int32_t step;

        while (cnt > 0)
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
