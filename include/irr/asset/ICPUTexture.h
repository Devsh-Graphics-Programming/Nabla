#ifndef __IRR_I_CPU_TEXTURE_H_INCLUDED__
#define __IRR_I_CPU_TEXTURE_H_INCLUDED__

#include <utility>
#include <algorithm>
#include <cassert>
#include "irr/core/Types.h"
#include "irr/asset/IAsset.h"
#include "irr/asset/CImageData.h"
#include "ITexture.h" // for ITexture::E_TEXTURE_COUNT ... this enum should be in global scope

namespace irr { namespace asset
{

class ICPUTexture : public IAsset
{
protected:
    uint32_t m_size[3];
    asset::E_FORMAT m_colorFormat;
    video::ITexture::E_TEXTURE_TYPE m_type;
    core::vector<asset::CImageData*> m_mipmaps;

    using IteratorType = typename decltype(m_mipmaps)::iterator;
    using ConstIteratorType = typename decltype(m_mipmaps)::const_iterator;

public:
    using RangeType = std::pair<IteratorType, IteratorType>;
    using ConstRangeType = std::pair<ConstIteratorType, ConstIteratorType>;

private:
    template<typename VectorRef>
    inline static ICPUTexture* create_impl(VectorRef&& _mipmaps)
    {
        if (!validateMipchain(_mipmaps))
            return nullptr;

        return new ICPUTexture(std::forward<decltype(_mipmaps)>(_mipmaps));
    }

public:
    inline static ICPUTexture* create(const core::vector<asset::CImageData*>& _mipmaps)
    {
        return create_impl(_mipmaps);
    }
    inline static ICPUTexture* create(core::vector<asset::CImageData*>&& _mipmaps)
    {
        return create_impl(std::move(_mipmaps));
    }
    template<typename Iter>
    inline static ICPUTexture* create(Iter _first, Iter _last)
    {
        return create(core::vector<asset::CImageData*>(_first, _last));
    }

    static bool validateMipchain(const core::vector<CImageData*>& _mipchain)
    {
        if (_mipchain.empty())
            return false;

        bool allUnknownFmt = true;
        E_FORMAT commonFmt = _mipchain.front()->getColorFormat();
        for (auto mip : _mipchain)
        {
            if (mip->getSupposedMipLevel() > 16u)
                return false;
            if (std::max_element(mip->getSliceMax(), mip->getSliceMax()+3)[0] > (0x10000>>mip->getSupposedMipLevel()))
                return false;
            if (commonFmt != EF_UNKNOWN)
            {
                if (mip->getColorFormat() != commonFmt && mip->getColorFormat() != EF_UNKNOWN)
                    return false;
            }
            else commonFmt = mip->getColorFormat();

            allUnknownFmt &= (mip->getColorFormat()==EF_UNKNOWN);
        }

        return !allUnknownFmt;
    }

protected:
    explicit ICPUTexture(const core::vector<asset::CImageData*>& _mipmaps) : m_mipmaps{_mipmaps}, m_colorFormat{EF_UNKNOWN}, m_type{video::ITexture::ETT_COUNT}
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
    explicit ICPUTexture(core::vector<asset::CImageData*>&& _mipmaps) : m_mipmaps{std::move(_mipmaps)}, m_colorFormat{EF_UNKNOWN}, m_type{ video::ITexture::ETT_COUNT }
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

    const core::vector<asset::CImageData*>& getMipmaps() const { return m_mipmaps; }

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

    inline asset::E_FORMAT getColorFormat() const { return m_colorFormat; }

    inline video::ITexture::E_TEXTURE_TYPE getType() const { return m_type; }

    inline const uint32_t* getSize() const { return m_size; }

private:
    inline void sortMipMaps()
    {
        std::sort(std::begin(m_mipmaps), std::end(m_mipmaps), 
            [](const asset::CImageData* _a, const asset::CImageData* _b) { return _a->getSupposedMipLevel() < _b->getSupposedMipLevel(); }
        );
    }
    inline void establishFmt()
    {
        m_colorFormat = (*std::find_if(m_mipmaps.begin(), m_mipmaps.end(), [](CImageData* mip) { return mip->getColorFormat() != EF_UNKNOWN; }))->getColorFormat();
    }
    inline void recalcSize()
    {
        m_size[0] = m_size[1] = m_size[2] = 1u;
        for (const asset::CImageData* img : m_mipmaps)
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