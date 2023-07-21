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

class ICPUImage final : public IImage, public IAsset
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
            clone_common(cp.get());

            if(regions && !regions->empty())
                cp->regions = core::make_refctd_dynamic_array<decltype(regions)>(*regions);

            cp->buffer = (_depth > 0u && buffer) ? core::smart_refctd_ptr_static_cast<ICPUBuffer>(buffer->clone(_depth-1u)) : buffer;

            return cp;
        }

        inline void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override
        {
            convertToDummyObject_common(referenceLevelsBelowToConvert);

			if (referenceLevelsBelowToConvert)
			if (buffer)
				buffer->convertToDummyObject(referenceLevelsBelowToConvert-1u);

			if (canBeConvertedToDummy())
				regions = nullptr;
        }

		_NBL_STATIC_INLINE_CONSTEXPR auto AssetType = ET_IMAGE;
		inline IAsset::E_TYPE getAssetType() const override { return AssetType; }

        virtual size_t conservativeSizeEstimate() const override
		{
			assert(regions);
			return sizeof(SCreationParams)+sizeof(void*)+sizeof(SBufferCopy)*regions->size();
		}

		virtual bool validateCopies(const SImageCopy* pRegionsBegin, const SImageCopy* pRegionsEnd, const ICPUImage* src) const
		{
			return validateCopies_template(pRegionsBegin, pRegionsEnd, src);
		}

		inline ICPUBuffer* getBuffer() 
		{
			assert(!isImmutable_debug());

			return buffer.get(); 
		}
		inline const auto* getBuffer() const { return buffer.get(); }

		inline core::SRange<const IImage::SBufferCopy> getRegions() const
		{
			if (regions)
				return {regions->begin(),regions->end()};
			return {nullptr,nullptr};
		}

		inline core::SRange<const IImage::SBufferCopy> getRegions(uint32_t mipLevel) const
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
		inline const IImage::SBufferCopy* getRegion(uint32_t mipLevel, const core::vectorSIMDu32& texelCoord) const
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
		inline auto wrapTextureCoordinate(uint32_t mipLevel, const core::vectorSIMDi32& texelCoord, const ISampler::E_TEXTURE_CLAMP wrapModes[3]) const
		{
			auto mipExtent = getMipSize(mipLevel);
			auto mipLastCoord = mipExtent-core::vector3du32_SIMD(1,1,1,1);
			return ICPUSampler::wrapTextureCoordinate(texelCoord,wrapModes,mipExtent,mipLastCoord);
		}


		//
		inline void* getTexelBlockData(const IImage::SBufferCopy* region, const core::vectorSIMDu32& inRegionCoord, core::vectorSIMDu32& outBlockCoord)
		{
			assert(!isImmutable_debug());

			auto localXYZLayerOffset = inRegionCoord/info.getDimension();
			outBlockCoord = inRegionCoord-localXYZLayerOffset*info.getDimension();
			return reinterpret_cast<uint8_t*>(buffer->getPointer())+region->getByteOffset(localXYZLayerOffset,region->getByteStrides(info));
		}
		inline const void* getTexelBlockData(const IImage::SBufferCopy* region, const core::vectorSIMDu32& inRegionCoord, core::vectorSIMDu32& outBlockCoord) const
		{
			return const_cast<typename std::decay<decltype(*this)>::type*>(this)->getTexelBlockData(region,inRegionCoord,outBlockCoord);
		}

		inline void* getTexelBlockData(uint32_t mipLevel, const core::vectorSIMDu32& boundedTexelCoord, core::vectorSIMDu32& outBlockCoord)
		{
			assert(!isImmutable_debug());

			// get region for coord
			const auto* region = getRegion(mipLevel,boundedTexelCoord);
			if (!region)
				return nullptr;
			//
			core::vectorSIMDu32 inRegionCoord(boundedTexelCoord);
			inRegionCoord -= core::vectorSIMDu32(region->imageOffset.x,region->imageOffset.y,region->imageOffset.z,region->imageSubresource.baseArrayLayer);
			return getTexelBlockData(region,inRegionCoord,outBlockCoord);
		}
		inline const void* getTexelBlockData(uint32_t mipLevel, const core::vectorSIMDu32& inRegionCoord, core::vectorSIMDu32& outBlockCoord) const
		{
			return const_cast<typename std::decay<decltype(*this)>::type*>(this)->getTexelBlockData(mipLevel,inRegionCoord,outBlockCoord);
		}


		//! regions will be copied and sorted
		inline bool setBufferAndRegions(core::smart_refctd_ptr<ICPUBuffer>&& _buffer, const core::smart_refctd_dynamic_array<IImage::SBufferCopy>& _regions)
		{
			assert(!isImmutable_debug());

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
			if(isImmutable_debug())
				return ((m_creationParams.usage & usage).value == usage.value);
			m_creationParams.usage = usage;
			return true;
		}

		inline bool addImageUsageFlags(core::bitflag<E_USAGE_FLAGS> usage)
		{
			if(isImmutable_debug())
				return ((m_creationParams.usage & usage).value == usage.value);
			m_creationParams.usage |= usage;
			return true;
		}

		bool canBeRestoredFrom(const IAsset* _other) const override
		{
			auto* other = static_cast<const ICPUImage*>(_other);
			if (info != other->info)
				return false;
			if (m_creationParams != other->m_creationParams)
				return false;
			if (!buffer->canBeRestoredFrom(other->buffer.get()))
				return false;

			return true;
		}

    protected:
		void restoreFromDummy_impl(IAsset* _other, uint32_t _levelsBelow) override
		{
			auto* other = static_cast<ICPUImage*>(_other);

			const bool restorable = willBeRestoredFrom(_other);

			if (restorable)
				std::swap(regions, other->regions);

			if (_levelsBelow)
				restoreFromDummy_impl_call(buffer.get(), other->buffer.get(), _levelsBelow - 1u);
		}

		bool isAnyDependencyDummy_impl(uint32_t _levelsBelow) const override
		{
			--_levelsBelow;
			return buffer->isAnyDependencyDummy(_levelsBelow);
		}

		ICPUImage(const SCreationParams& _params) : IImage(_params)
		{
		}

		virtual ~ICPUImage() = default;
		
		
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


