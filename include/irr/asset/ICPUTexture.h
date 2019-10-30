#ifndef __IRR_I_CPU_TEXTURE_H_INCLUDED__
#define __IRR_I_CPU_TEXTURE_H_INCLUDED__

#include <utility>
#include <algorithm>
#include <cassert>

namespace irr { namespace asset
{

class ICPUTexture : public IAsset, public IDescriptor
{
public:
    using RangeType = std::pair<IteratorType, IteratorType>;
    using ConstRangeType = std::pair<ConstIteratorType, ConstIteratorType>;

private:
    template<typename VectorRef>
    inline static ICPUTexture* create_impl(VectorRef&& _textureRanges, const std::string& _srcFileName, video::ITexture::E_TEXTURE_TYPE _Type)
    {
        if (!validateMipchain(_textureRanges, _Type))
            return nullptr;

        return new ICPUTexture(std::forward<decltype(_textureRanges)>(_textureRanges), _srcFileName, _Type);
    }

public:
    inline static ICPUTexture* create(const core::vector<asset::CImageData*>& _textureRanges, const std::string& _srcFileName, video::ITexture::E_TEXTURE_TYPE _Type = video::ITexture::ETT_COUNT)
    {
        return create_impl(_textureRanges, _srcFileName, _Type);
    }
    inline static ICPUTexture* create(core::vector<asset::CImageData*>&& _textureRanges, const std::string& _srcFileName, video::ITexture::E_TEXTURE_TYPE _Type = video::ITexture::ETT_COUNT)
    {
        return create_impl(std::move(_textureRanges), _srcFileName, _Type);
    }
    template<typename Iter>
    inline static ICPUTexture* create(Iter _first, Iter _last, const std::string& _srcFileName, video::ITexture::E_TEXTURE_TYPE _Type = video::ITexture::ETT_COUNT)
    {
        return create(core::vector<asset::CImageData*>(_first, _last), _srcFileName, _Type);
    }


protected:
    explicit ICPUTexture(const core::vector<asset::CImageData*>& _textureRanges, const std::string& _srcFileName, video::ITexture::E_TEXTURE_TYPE _Type) : m_textureRanges{ _textureRanges }, m_colorFormat{EF_UNKNOWN}, m_type{_Type}, m_name{_srcFileName}
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
    explicit ICPUTexture(core::vector<asset::CImageData*>&& _textureRanges, const std::string& _srcFileName, video::ITexture::E_TEXTURE_TYPE _Type) : m_textureRanges{std::move(_textureRanges)}, m_colorFormat{EF_UNKNOWN}, m_type{_Type}, m_name{_srcFileName}
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
		const uint32_t mipDimensions[video::ITexture::ETT_COUNT+1u] = {1u,2u,3u,1u,2u,2u,2u,0u};

        memset(m_minReqBaseLvlSz, 0, sizeof(m_minReqBaseLvlSz));
        for (CImageData* _range : m_textureRanges)
        {
            for (uint32_t d = 0u; d < 3u; ++d)
            {
				uint32_t extent = _range->getSliceMax()[d];
				if (d<mipDimensions[m_type])
					extent = extent << _range->getSupposedMipLevel();

                if (m_minReqBaseLvlSz[d] < extent)
                    m_minReqBaseLvlSz[d] = extent;
            }
        }

		if (m_type == video::ITexture::ETT_CUBE_MAP || m_type == video::ITexture::ETT_CUBE_MAP_ARRAY)
			m_minReqBaseLvlSz[0u] = m_minReqBaseLvlSz[1] = std::max(m_minReqBaseLvlSz[0u],m_minReqBaseLvlSz[1]);
    }
};

}}//irr::asset

#endif//__IRR_I_CPU_TEXTURE_H_INCLUDED__