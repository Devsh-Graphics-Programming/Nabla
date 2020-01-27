// Copyright (C) 2017- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_CPU_IMAGE_H_INCLUDED__
#define __I_CPU_IMAGE_H_INCLUDED__

#include "irr/core/core.h"

#include "irr/asset/IAsset.h"
#include "irr/asset/ICPUBuffer.h"
#include "irr/asset/IImage.h"

namespace irr
{
namespace asset
{

class ICPUImage final : public IImage, public IAsset
{
	public:
		inline static core::smart_refctd_ptr<ICPUImage> create(SCreationParams&& _params)
		{
			if (!validateCreationParameters(_params))
				return nullptr;

			return core::smart_refctd_ptr<ICPUImage>(new ICPUImage(std::move(_params)), core::dont_grab);
		}

        core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
        {
            auto par = params;
            auto cp = core::smart_refctd_ptr<ICPUImage>(new ICPUImage(std::move(par)), core::dont_grab);
            clone_common(cp.get());

            cp->regions = regions;
            cp->buffer = (_depth > 0u && buffer) ? core::smart_refctd_ptr_static_cast<ICPUBuffer>(buffer->clone(_depth-1u)) : buffer;

            return cp;
        }

        inline void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override
        {
            if (isDummyObjectForCacheAliasing)
                return;
            convertToDummyObject_common(referenceLevelsBelowToConvert);

			if (referenceLevelsBelowToConvert)
			if (buffer)
				buffer->convertToDummyObject(referenceLevelsBelowToConvert-1u);
            regions = nullptr;
        }

        inline IAsset::E_TYPE getAssetType() const override { return IAsset::ET_IMAGE; }

        virtual size_t conservativeSizeEstimate() const override
		{
			assert(regions);
			return sizeof(SCreationParams)+sizeof(void*)+sizeof(SBufferCopy)*regions->size();
		}

		virtual bool validateCopies(const SImageCopy* pRegionsBegin, const SImageCopy* pRegionsEnd, const ICPUImage* src)
		{
			return validateCopies_template(pRegionsBegin, pRegionsEnd, src);
		}

		inline auto* getBuffer() { return buffer.get(); }
		inline const auto* getBuffer() const { return buffer.get(); }

		inline core::SRange<const IImage::SBufferCopy> getRegions() const
		{
			if (regions)
				return {regions->begin(),regions->end()};
			return {nullptr,nullptr};
		}

		//! regions will be copied and sorted
		inline bool setBufferAndRegions(core::smart_refctd_ptr<ICPUBuffer>&& _buffer, const core::smart_refctd_dynamic_array<IImage::SBufferCopy>& _regions)
		{
			if (!IImage::validateCopies(_regions->begin(),_regions->end(),_buffer.get()))
				return false;
		
			buffer = std::move(_buffer);
			regions = _regions;
			sortRegionsByMipMapLevel();
			return true;
		}

		/*
        inline void* getRowPointer(const SBufferCopy& region, uint32_t relMip, uint32_t relZ, uint32_t relY)
		{
            return const_cast<void*>(getRowPointer(region,relMip,relZ,relY));
		}
        inline const void* getRowPointer(const SBufferCopy& region, uint32_t relMip, uint32_t relZ, uint32_t relY) const
        {
            if (asset::isBlockCompressionFormat(getColorFormat()))
                return nullptr;

            if (row<minCoord[0]||row>=maxCoord[0])
                return nullptr;
            if (slice<minCoord[1]||slice>=maxCoord[1])
                return nullptr;

            size_t size[3] = {maxCoord[0]-minCoord[0],maxCoord[1]-minCoord[1],maxCoord[2]-minCoord[2]};
            row     -= minCoord[0];
            slice   -= minCoord[1];
            return reinterpret_cast<uint8_t*>(data)+(slice*size[1]+row)*getPitchIncludingAlignment();
        }
		

        //!
        inline uint32_t getPitchIncludingAlignment() const
        {
            if (isBlockCompressionFormat(getColorFormat()))
                return 0; //special error val

			auto lineBytes = getBytesPerPixel() * (maxCoord[0]-minCoord[0]);
			assert(lineBytes.getNumerator()%lineBytes.getDenominator() == 0u);
            return (lineBytes.getNumerator()/lineBytes.getDenominator()+unpackAlignment-1)&(~(unpackAlignment-1u));
        }
*/

    protected:
		ICPUImage(SCreationParams&& _params) : IImage(std::move(_params))
		{
		}

		virtual ~ICPUImage() = default;
		
		
		core::smart_refctd_ptr<asset::ICPUBuffer>				buffer;
		core::smart_refctd_dynamic_array<IImage::SBufferCopy>	regions;

	private:
		inline void sortRegionsByMipMapLevel()
		{
			std::sort(regions->begin(), regions->end(), [](const IImage::SBufferCopy& _a, const IImage::SBufferCopy& _b)
				{
					if (_a.imageSubresource.mipLevel==_b.imageSubresource.mipLevel)
					{
						if (_a.imageSubresource.baseArrayLayer==_b.imageSubresource.baseArrayLayer)
						{
							if (_a.imageOffset.z==_b.imageOffset.z)
							{
								if (_a.imageOffset.y==_b.imageOffset.y)
									return _a.imageOffset.x<_b.imageOffset.x;
								else
									return _a.imageOffset.y<_b.imageOffset.y;
							}
							else
								return _a.imageOffset.z<_b.imageOffset.z;
						}
						else
							return _a.imageSubresource.baseArrayLayer<_b.imageSubresource.baseArrayLayer;
					}
					else
						return _a.imageSubresource.mipLevel<_b.imageSubresource.mipLevel;
				}
			);
		}
};

} // end namespace video
} // end namespace irr

#endif


