// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_IMAGE_H_INCLUDED_
#define _NBL_ASSET_I_CPU_IMAGE_H_INCLUDED_

#include "nbl/core/declarations.h"

#include "nbl/asset/IAsset.h"
#include "nbl/asset/ICPUBuffer.h"
#include "nbl/asset/IImage.h"
#include "nbl/asset/ICPUSampler.h"

namespace nbl::asset
{

class NBL_API2 ICPUImage final : public IImage, public IPreHashed
{
	public:
		inline static core::smart_refctd_ptr<ICPUImage> create(const SCreationParams& _params)
		{
			if (!validateCreationParameters(_params))
				return nullptr;

			return core::smart_refctd_ptr<ICPUImage>(new ICPUImage(_params), core::dont_grab);
		}

        core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
        {
            auto par = m_creationParams;
            auto cp = core::smart_refctd_ptr<ICPUImage>(new ICPUImage(std::move(par)), core::dont_grab);

            if(regions && !regions->empty())
                cp->regions = core::make_refctd_dynamic_array<decltype(regions)>(*regions);

			if (_depth > 0u && buffer)
				cp->buffer = core::smart_refctd_ptr_static_cast<ICPUBuffer>(buffer->clone(_depth-1u));
			else
				cp->buffer = buffer;

			cp->setContentHash(getContentHash());
            return cp;
        }

		constexpr static inline auto AssetType = ET_IMAGE;
		inline IAsset::E_TYPE getAssetType() const override { return AssetType; }

		// Do not report buffer as dependant, as we will simply drop it instead of discarding its contents!
		inline size_t getDependantCount() const override {return 0;}

		core::blake3_hash_t computeContentHash() const override;

		// Having regions specififed to upload is optional! So to have content missing we must have regions but no buffer content
		inline bool missingContent() const override
		{
			return regions && !regions->empty() && (!buffer || buffer->missingContent());
		}

		virtual bool validateCopies(const SImageCopy* pRegionsBegin, const SImageCopy* pRegionsEnd, const ICPUImage* src) const
		{
			return validateCopies_template(pRegionsBegin, pRegionsEnd, src);
		}

		inline ICPUBuffer* getBuffer() 
		{
			assert(isMutable());

			return buffer.get(); 
		}
		inline const auto* getBuffer() const { return buffer.get(); }

		inline std::span<const IImage::SBufferCopy> getRegions() const
		{
			if (regions)
				return {regions->begin(),regions->end()};
			return {};
		}

		inline std::span<const IImage::SBufferCopy> getRegions(uint32_t mipLevel) const
		{
			const IImage::SBufferCopy dummy = { 0ull,0u,0u,{static_cast<E_ASPECT_FLAGS>(0u),mipLevel,0u,0u},{},{} };
			auto begin = std::lower_bound(regions->begin(),regions->end(),dummy,mip_order_t());
			auto end = std::upper_bound(regions->begin(),regions->end(),dummy,mip_order_t());
			return {begin,end};
		}
		inline auto getRegionArray() const
		{
			using immutable_refctd_array_t = core::refctd_dynamic_array<IImage::SBufferCopy>;
			return core::smart_refctd_ptr<immutable_refctd_array_t>(reinterpret_cast<immutable_refctd_array_t*>(regions.get()));
		}
		inline auto getRegionArray() { return regions; }

		// `texelCoord=(xTexelPos,yTexelPos,zTexelPos,imageLayer)`
		inline const IImage::SBufferCopy* getRegion(uint32_t mipLevel, const hlsl::uint32_t4& texelCoord) const
		{
			auto mip = getRegions(mipLevel);
			auto found = std::find_if(std::reverse_iterator(mip.end()),std::reverse_iterator(mip.begin()),
				[&texelCoord](const IImage::SBufferCopy& region)
				{ // we can simdify this in the future
					if (region.imageSubresource.baseArrayLayer>texelCoord.w)
						return false;
					if (texelCoord.w>=region.imageSubresource.baseArrayLayer+region.imageSubresource.layerCount)
						return false;

					bool retval = true;
					for (auto i=0; i<3; i++)
					{
						const auto _min = (&region.imageOffset.x)[i];
						const auto _max = _min+(&region.imageExtent.width)[i];
						retval = retval && texelCoord[i]>=_min && texelCoord[i]<_max;
					}
					return retval;
				}
			);
			if (found!=std::reverse_iterator(mip.begin()))
				return &(*found);
			return nullptr;
		}

		//
		inline auto wrapTextureCoordinate(uint32_t mipLevel, const hlsl::int32_t4& texelCoord, const ISampler::E_TEXTURE_CLAMP wrapModes[3]) const
		{
			auto mipExtent = getMipSize(mipLevel);
			auto mipLastCoord = mipExtent- hlsl::uint32_t3(1,1,1);
			return ICPUSampler::wrapTextureCoordinate(texelCoord,wrapModes,mipExtent,mipLastCoord);
		}


		//
		inline void* getTexelBlockData(const IImage::SBufferCopy* region, const hlsl::uint32_t3& inRegionCoord, hlsl::uint32_t3& outBlockCoord)
		{
			assert(isMutable());

			auto localXYZLayerOffset = inRegionCoord/info.getDimension();
			outBlockCoord = inRegionCoord-localXYZLayerOffset*info.getDimension();
			return reinterpret_cast<uint8_t*>(buffer->getPointer())+region->getByteOffset(localXYZLayerOffset,region->getByteStrides(info));
		}
		inline const void* getTexelBlockData(const IImage::SBufferCopy* region, const hlsl::uint32_t3& inRegionCoord, hlsl::uint32_t3& outBlockCoord) const
		{
			return const_cast<typename std::decay<decltype(*this)>::type*>(this)->getTexelBlockData(region,inRegionCoord,outBlockCoord);
		}

		inline void* getTexelBlockData(uint32_t mipLevel, const hlsl::uint32_t4& boundedTexelCoord, hlsl::uint32_t3& outBlockCoord)
		{
			assert(isMutable());

			// get region for coord
			const auto* region = getRegion(mipLevel,boundedTexelCoord);
			if (!region)
				return nullptr;
			//
			hlsl::uint32_t4 inRegionCoord(boundedTexelCoord);
			inRegionCoord -= hlsl::uint32_t4(region->imageOffset.x,region->imageOffset.y,region->imageOffset.z,region->imageSubresource.baseArrayLayer);
			return getTexelBlockData(region,inRegionCoord,outBlockCoord);
		}
		inline const void* getTexelBlockData(uint32_t mipLevel, const hlsl::uint32_t3& inRegionCoord, hlsl::uint32_t3& outBlockCoord) const
		{
			return const_cast<typename std::decay<decltype(*this)>::type*>(this)->getTexelBlockData(mipLevel,inRegionCoord,outBlockCoord);
		}


		//! regions will be copied and sorted
		inline bool setBufferAndRegions(core::smart_refctd_ptr<ICPUBuffer>&& _buffer, const core::smart_refctd_dynamic_array<IImage::SBufferCopy>& _regions)
		{
			assert(isMutable());

			if (!IImage::validateCopies(_regions->begin(),_regions->end(),_buffer.get()))
			{
				assert(false);
				return false;
			}
		
			buffer = std::move(_buffer);
			regions = _regions;
			std::sort(regions->begin(),regions->end(),mip_order_t());
			addImageUsageFlags(EUF_TRANSFER_DST_BIT);
			return true;
		}
		
		inline core::bitflag<E_USAGE_FLAGS> getImageUsageFlags() const
		{
			return m_creationParams.usage;
		}

		inline bool setImageUsageFlags(core::bitflag<E_USAGE_FLAGS> usage)
		{
			if(!isMutable())
				return ((m_creationParams.usage & usage).value == usage.value);
			m_creationParams.usage = usage;
			return true;
		}

		inline bool addImageUsageFlags(core::bitflag<E_USAGE_FLAGS> usage)
		{
			if(!isMutable())
				return ((m_creationParams.usage & usage).value == usage.value);
			m_creationParams.usage |= usage;
			return true;
		}

    protected:
		inline ICPUImage(const SCreationParams& _params) : IImage(_params) {}
		virtual ~ICPUImage() = default;
		
		inline IAsset* getDependant_impl(const size_t ix) override {return buffer.get();}

		inline void discardContent_impl() override
		{
			buffer = nullptr;
		}
		
		// TODO: maybe we shouldn't make a single buffer back all regions?
		core::smart_refctd_ptr<asset::ICPUBuffer>				buffer;
		core::smart_refctd_dynamic_array<IImage::SBufferCopy>	regions;

	private:
		struct mip_order_t
		{
			inline bool operator()(const IImage::SBufferCopy& _a, const IImage::SBufferCopy& _b)
			{
				return _a.imageSubresource.mipLevel < _b.imageSubresource.mipLevel;
			}
		};
};

} // end namespace nbl::asset

#endif


