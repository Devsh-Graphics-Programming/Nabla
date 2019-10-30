// Copyright (C) 2017- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_CPU_IMAGE_H_INCLUDED__
#define __I_CPU_IMAGE_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"

#include "irr/asset/IImage.h"
#include "irr/asset/IAsset.h"

namespace irr
{
namespace asset
{

class ICPUImage final : public IImage, public IAsset
{
	public:
		inline static core::smart_refctd_ptr<ICPUTexture> create()
		{
			return nullptr;
		}

        inline void convertToDummyObject() override
        {
            regions = nullptr;
        }

        inline E_TYPE getAssetType() const override { return IAsset::ET_IMAGE; }

        virtual size_t conservativeSizeEstimate() const override
		{
			return sizeof(IImage)-sizeof(IDescriptor)+;
		}


		inline auto* getBuffer() { return buffer.get(); }
		inline const auto* getBuffer() const { return buffer.get(); }

		inline void setBuffer(core::smart_refctd_ptr<core::ICPUBuffer>&& _buffer) { buffer = _buffer; }

		/*
        inline const void* getSliceRowPointer_helper(const SBufferCopy& ) const
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

        //!
        inline void* getSliceRowPointer(const uint32_t& slice, const uint32_t& row)
        {
            return const_cast<void*>(getSliceRowPointer_helper(slice,row)); // I know what I'm doing
        }
        inline const void* getSliceRowPointer(const uint32_t& slice, const uint32_t& row) const
        {
            return getSliceRowPointer_helper(slice,row);
        }
*/

    protected:
		ICPUImage(	core::smart_refctd_ptr<asset::ICPUBuffer>&& _buffer,
					core::smart_refctd_dynamic_array<IImage::SBufferCopy>&& _regions,
					E_IMAGE_CREATE_FLAGS _flags,
					E_IMAGE_TYPE _type,
					E_FORMAT _format,
					const VkExtent3D& _extent,
					uint32_t _mipLevels=0u,
					uint32_t _arrayLayers=1u,
					E_SAMPLE_COUNT_FLAGS _samples=ESMF_1_BIT) :
						IImage(_flags,_type,_format,_extent,_mipLevels,_arrayLayers,_samples),
						buffer(_buffer), regions(_regions)
		{
		}

		virtual ~ICPUImage() = default;
		
		
		core::smart_refctd_ptr<asset::ICPUBuffer>				buffer;
		core::smart_refctd_dynamic_array<IImage::SBufferCopy>	regions;

	private:
		inline void sortRegionsByMipMapLevel()
		{
			std::sort(std::begin(m_textureRanges), std::end(m_textureRanges),
				[](const asset::CImageData* _a, const asset::CImageData* _b) { return _a->getSupposedMipLevel() < _b->getSupposedMipLevel(); }
			);
		}
};

} // end namespace video
} // end namespace irr

#endif


