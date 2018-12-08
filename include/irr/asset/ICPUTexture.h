#ifndef __IRR_I_CPU_TEXTURE_H_INCLUDED__
#define __IRR_I_CPU_TEXTURE_H_INCLUDED__

#include <utility>
#include <algorithm>
#include <cassert>
#include "irr/core/Types.h"
#include "irr/asset/IAsset.h"
#include "CImageData.h"
#include "ITexture.h" // for ITexture::E_TEXTURE_COUNT ... this enum should be in global scope

namespace irr { namespace asset
{

class ICPUTexture : public IAsset
{
protected:
    uint32_t m_size[3];
    video::E_FORMAT m_colorFormat;
    video::ITexture::E_TEXTURE_TYPE m_type;
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
        video::E_FORMAT colorFmt = _mipmaps.front()->getColorFormat();
        for (const video::CImageData* img : _mipmaps)
        {
            const uint32_t lvl = img->getSupposedMipLevel();
            check[lvl] = true;
            if (lvl > 16u)
                return nullptr;

            const video::E_FORMAT fmt = img->getColorFormat();
            if (fmt != video::EF_UNKNOWN)
                allUnknown = false;
            if (fmt != colorFmt && fmt != video::EF_UNKNOWN)
                return nullptr;
            if (colorFmt == video::EF_UNKNOWN)
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
    explicit ICPUTexture(const core::vector<video::CImageData*>& _mipmaps) : m_mipmaps{_mipmaps}, m_colorFormat{video::EF_UNKNOWN}, m_type{video::ITexture::ETT_COUNT}
    {
        for (const auto& mm : m_mipmaps)
            mm->grab();
        if (m_mipmaps.size())
        {
            recalcSize();
            sortMipMaps();
            establishFmt();
            establishType();
        }
    }
    explicit ICPUTexture(core::vector<video::CImageData*>&& _mipmaps) : m_mipmaps{std::move(_mipmaps)}, m_colorFormat{video::EF_UNKNOWN}, m_type{ video::ITexture::ETT_COUNT }
    {
        for (const auto& mm : m_mipmaps)
            mm->grab();
        if (m_mipmaps.size())
        {
            recalcSize();
            sortMipMaps();
            establishFmt();
            establishType();
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

    inline video::E_FORMAT getColorFormat() const { return m_colorFormat; }

    inline video::ITexture::E_TEXTURE_TYPE getType() const { return m_type; }

    inline const uint32_t* getSize() const { return m_size; }

private:
    inline void sortMipMaps()
    {
        std::sort(std::begin(m_mipmaps), std::end(m_mipmaps), 
            [](const video::CImageData* _a, const video::CImageData* _b) { return _a->getSupposedMipLevel() < _b->getSupposedMipLevel(); }
        );
    }
    inline void establishFmt()
    {
        video::E_FORMAT fmt = m_mipmaps[0]->getColorFormat();
        for (uint32_t i = 1u; i < m_mipmaps.size(); ++i)
        {
            if (fmt != m_mipmaps[i]->getColorFormat())
            {
                fmt = video::EF_UNKNOWN;
                break;
            }
        }
        m_colorFormat = fmt;
    }
    inline void recalcSize()
    {
        m_size[0] = m_size[1] = m_size[2] = 1u;
        for (const video::CImageData* img : m_mipmaps)
        {
            for (uint32_t i = 0u; i < 3u; ++i)
            {
                const uint32_t dimSz = img->getSliceMax()[i];
                if (m_size[i] < dimSz)
                    m_size[i] = dimSz;
            }
        }
    }
    //! Assumes that m_mipmaps is already sorted by mip-lvl and size calculated
    // needs to be reworked for sparse textures
    inline void establishType()
    {
        if (m_size[2] > 1u)
        {
            // with this little info I literally can't guess if you want a cubemap!
            if (m_mipmaps.size() > 1 && m_mipmaps.front()->getSliceMax()[2] == m_mipmaps.back()->getSliceMax()[2])
                m_type = video::ITexture::ETT_2D_ARRAY;
            else
                m_type = video::ITexture::ETT_3D;
        }
        else if (m_size[1] > 1u)
        {
            if (m_mipmaps.size() > 1 && m_mipmaps.front()->getSliceMax()[1] == m_mipmaps.back()->getSliceMax()[1])
                m_type = video::ITexture::ETT_1D_ARRAY;
            else
                m_type = video::ITexture::ETT_2D;
        }
        else
        {
            m_type = video::ITexture::ETT_2D; //should be ETT_1D but 2D is default since forever
        }
    }
};

}}//irr::asset

#endif//__IRR_I_CPU_TEXTURE_H_INCLUDED__