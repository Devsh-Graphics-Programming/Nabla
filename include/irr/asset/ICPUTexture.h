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
    uint32_t m_minReqBaseLvlSz[3];
    core::vector<asset::CImageData*> m_textureRanges;
    asset::E_FORMAT m_colorFormat;
    video::ITexture::E_TEXTURE_TYPE m_type;

    using IteratorType = typename decltype(m_textureRanges)::iterator;
    using ConstIteratorType = typename decltype(m_textureRanges)::const_iterator;

public:
    using RangeType = std::pair<IteratorType, IteratorType>;
    using ConstRangeType = std::pair<ConstIteratorType, ConstIteratorType>;

private:
    template<typename VectorRef>
    inline static ICPUTexture* create_impl(VectorRef&& _textureRanges, video::ITexture::E_TEXTURE_TYPE _Type)
    {
        if (!validateMipchain(_textureRanges, _Type))
            return nullptr;

        return new ICPUTexture(std::forward<decltype(_textureRanges)>(_textureRanges), _Type);
    }

public:
    inline static ICPUTexture* create(const core::vector<asset::CImageData*>& _textureRanges, video::ITexture::E_TEXTURE_TYPE _Type = video::ITexture::ETT_COUNT)
    {
        return create_impl(_textureRanges, _Type);
    }
    inline static ICPUTexture* create(core::vector<asset::CImageData*>&& _textureRanges, video::ITexture::E_TEXTURE_TYPE _Type = video::ITexture::ETT_COUNT)
    {
        return create_impl(std::move(_textureRanges), _Type);
    }
    template<typename Iter>
    inline static ICPUTexture* create(Iter _first, Iter _last, video::ITexture::E_TEXTURE_TYPE _Type = video::ITexture::ETT_COUNT)
    {
        return create(core::vector<asset::CImageData*>(_first, _last), _Type);
    }

    static bool validateMipchain(const core::vector<CImageData*>& _textureRanges, video::ITexture::E_TEXTURE_TYPE _Type)
    {
        if (_textureRanges.empty())
            return false;

        bool allUnknownFmt = true;
        E_FORMAT commonFmt = _textureRanges.front()->getColorFormat();
        for (auto _range : _textureRanges)
        {
            if (_range->getSupposedMipLevel() > 16u)
                return false;
            if (std::max_element(_range->getSliceMax(), _range->getSliceMax()+3)[0] > (0x10000u>> _range->getSupposedMipLevel()))
                return false;

			switch (_Type)
			{
				case video::ITexture::ETT_1D:
					if (_range->getSliceMax()[1] > 1u)
						return false;
					_IRR_FALLTHROUGH;
				case video::ITexture::ETT_1D_ARRAY:
					_IRR_FALLTHROUGH;
				case video::ITexture::ETT_2D:
					if (_range->getSliceMax()[2] > 1u)
						return false;
					break;
				case video::ITexture::ETT_CUBE_MAP_ARRAY:
					if (_range->getSliceMax()[2]%6u != 0u)
						return false;
					_IRR_FALLTHROUGH;
				case video::ITexture::ETT_CUBE_MAP:
					if (_range->getSliceMax()[2] > 6u)
						return false;
					break;
				default: //	type unknown, or 3D format
					break;
			}

            if (commonFmt != EF_UNKNOWN)
            {
                if (_range->getColorFormat() != commonFmt && _range->getColorFormat() != EF_UNKNOWN)
                    return false;
            }
            else commonFmt = _range->getColorFormat();

            allUnknownFmt &= (_range->getColorFormat()==EF_UNKNOWN);
        }

        return !allUnknownFmt;
    }

protected:
    explicit ICPUTexture(const core::vector<asset::CImageData*>& _textureRanges, video::ITexture::E_TEXTURE_TYPE _Type) : m_textureRanges{ _textureRanges }, m_colorFormat{EF_UNKNOWN}, m_type{_Type}
    {
        for (const auto& mm : m_textureRanges)
            mm->grab();
        if (m_textureRanges.size())
        {
            recalcSize();
			sortRangesByMipMapLevel();
            establishFmt();
            establishType();
            establishMinBaseLevelSize();
        }
    }
    explicit ICPUTexture(core::vector<asset::CImageData*>&& _textureRanges, video::ITexture::E_TEXTURE_TYPE _Type) : m_textureRanges{std::move(_textureRanges)}, m_colorFormat{EF_UNKNOWN}, m_type{_Type}
    {
        for (const auto& mm : m_textureRanges)
            mm->grab();
        if (m_textureRanges.size())
        {
            recalcSize();
			sortRangesByMipMapLevel();
            establishFmt();
            establishType();
            establishMinBaseLevelSize();
        }
    }

    virtual ~ICPUTexture()
    {
        for (const auto& mm : m_textureRanges)
            mm->drop();
    }

public:
    virtual void convertToDummyObject() override {}

    virtual E_TYPE getAssetType() const override { return IAsset::ET_IMAGE; }

    virtual size_t conservativeSizeEstimate() const override
    {
        return getCacheKey().length()+1u;
    }

    const core::vector<asset::CImageData*>& getRanges() const { return m_textureRanges; }

    //! Finds range of images making up _mipLvl miplevel or higher if _mipLvl miplevel is not present. 
    //! @returns {end(), end()} if _mipLvl > highest present mip level.
    inline RangeType getMipMap(uint32_t _mipLvl)
    {
        if (_mipLvl > getHighestMip())
            return { m_textureRanges.end(), m_textureRanges.end()};
        IteratorType l = m_textureRanges.begin(), it;
        int32_t cnt = m_textureRanges.size();
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
        if (l == m_textureRanges.end())
            return std::make_pair(l, l);
        RangeType rng;
        rng.first = rng.second = l;
        while (rng.second != m_textureRanges.end() && (*rng.second)->getSupposedMipLevel() == (*rng.first)->getSupposedMipLevel())
            ++rng.second;
        return rng;
    }
    //! @returns {end(), end()} if _mipLvl level mipmap is not present.
    inline ConstRangeType getMipMap(uint32_t _mipLvl) const
    {
        return const_cast<ICPUTexture*>(this)->getMipMap(_mipLvl);
    }

    inline uint32_t getHighestMip() const { return m_textureRanges.back()->getSupposedMipLevel(); }

    inline asset::E_FORMAT getColorFormat() const { return m_colorFormat; }

    inline video::ITexture::E_TEXTURE_TYPE getType() const { return m_type; }

    inline const uint32_t* getSize() const { return m_size; }

    inline const uint32_t* getBaseLevelSizeHint() const { return m_minReqBaseLvlSz; }

private:
    inline void sortRangesByMipMapLevel()
    {
        std::sort(std::begin(m_textureRanges), std::end(m_textureRanges),
            [](const asset::CImageData* _a, const asset::CImageData* _b) { return _a->getSupposedMipLevel() < _b->getSupposedMipLevel(); }
        );
    }
    inline void establishFmt()
    {
        m_colorFormat = (*std::find_if(m_textureRanges.begin(), m_textureRanges.end(), [](CImageData* mip) { return mip->getColorFormat() != EF_UNKNOWN; }))->getColorFormat();
    }
    inline void recalcSize()
    {
        m_size[0] = m_size[1] = m_size[2] = 1u;
        for (const asset::CImageData* img : m_textureRanges)
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
		if (m_type < video::ITexture::ETT_COUNT) // we know the type
			return;

        if (m_size[2] > 1u)
        {
            // with this little info I literally can't guess if you want a cubemap!
            if (m_textureRanges.size() > 1 && m_textureRanges.front()->getSliceMax()[2] == m_textureRanges.back()->getSliceMax()[2]) // TODO: Need better test, 2d array with mip-map chain will fail
                m_type = video::ITexture::ETT_2D_ARRAY;
            else
                m_type = video::ITexture::ETT_3D;
        }
        else if (m_size[1] > 1u)
        {
            if (m_textureRanges.size() > 1 && m_textureRanges.front()->getSliceMax()[1] == m_textureRanges.back()->getSliceMax()[1]) // TODO: Need better test, 2d array with mip-map chain will fail
                m_type = video::ITexture::ETT_1D_ARRAY;
            else
                m_type = video::ITexture::ETT_2D;
        }
        else
        {
            m_type = video::ITexture::ETT_2D; //should be ETT_1D but 2D is default since forever
        }
    }
    inline void establishMinBaseLevelSize()
    {
        memset(m_minReqBaseLvlSz, 0, sizeof(m_minReqBaseLvlSz));
        for (CImageData* _range : m_textureRanges)
        {
            for (uint32_t d = 0u; d < 3u; ++d)
            {
                const uint32_t extent = _range->getSliceMax()[d] << _range->getSupposedMipLevel();
                if (m_minReqBaseLvlSz[d] < extent)
                    m_minReqBaseLvlSz[d] = extent;
            }
        }
    }
};

}}//irr::asset

#endif//__IRR_I_CPU_TEXTURE_H_INCLUDED__