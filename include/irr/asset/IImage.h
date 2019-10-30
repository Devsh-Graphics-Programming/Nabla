// Copyright (C) 2017- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_IMAGE_H_INCLUDED__
#define __I_IMAGE_H_INCLUDED__

#include "irr/asset/format/EFormat.h"
#include "irr/asset/IDescriptor.h"

namespace irr
{
namespace asset
{
	
//placeholder until we configure Vulkan SDK
typedef struct VkOffset3D {
	uint32_t	x;
	uint32_t	y;
	uint32_t	z;
} VkOffset3D; //depr
typedef struct VkExtent3D {
	uint32_t	width;
	uint32_t	height;
	uint32_t	depth;
} VkExtent3D; //depr

// common
#ifdef _IRR_DEBUG // TODO: When Vulkan comes
	// check buffer contains all regions
	// check regions don't overlap
#endif
// any command
#ifdef _IRR_DEBUG // TODO: When Vulkan comes
	// dst image only has one MSAA sample
	// check regions contained in dstImage
#endif
// GPU command
#ifdef _IRR_DEBUG // TODO: When Vulkan comes
	// image offset and extent must respect granularity requirements
	// buffer has memory bound (with sparse exceptions)
	// check buffer has transfer usage flag
	// format features of dstImage contain transfer dst bit
	// check regions contained in dstImage
	// dst image not created subsampled
#endif
class IImage : public IDescriptor
{
	public:
		enum E_IMAGE_ASPECT_FLAGS
		{
			EIAF_COLOR_BIT			= 0x1u << 0u,
			EIAF_DEPTH_BIT			= 0x1u << 1u,
			EIAF_STENCIL_BIT		= 0x1u << 2u,
			EIAF_METADATA_BIT		= 0x1u << 3u,
			EIAF_PLANE_0_BIT		= 0x1u << 4u,
			EIAF_PLANE_1_BIT		= 0x1u << 5u,
			EIAF_PLANE_2_BIT		= 0x1u << 6u,
			EIAF_MEMORY_PLANE_0_BIT	= 0x1u << 7u,
			EIAF_MEMORY_PLANE_1_BIT	= 0x1u << 8u,
			EIAF_MEMORY_PLANE_2_BIT	= 0x1u << 9u,
			EIAF_MEMORY_PLANE_3_BIT	= 0x1u << 10u
		};
		enum E_IMAGE_CREATE_FLAGS : uint32_t
		{
			//! irrelevant now - no support for sparse or aliased resources
			EICF_SPARSE_BINDING_BIT						= 0x1u << 0u,
			EICF_SPARSE_RESIDENCY_BIT					= 0x1u << 1u,
			EICF_SPARSE_ALIASED_BIT						= 0x1u << 2u,
			//! irrelevant now - no support for planar images
			EICF_MUTABLE_FORMAT_BIT						= 0x1u << 3u,
			//! whether can fashion a cubemap out of the image
			EICF_CUBE_COMPATIBLE_BIT					= 0x1u << 4u,
			//! whether can fashion a 2d array texture out of the image
			EICF_2D_ARRAY_COMPATIBLE_BIT				= 0x1u << 5u,
			//! irrelevant now - we don't support device groups
			EICF_SPLIT_INSTANCE_BIND_REGIONS_BIT		= 0x1u << 6u,
			//! whether can view a block compressed texture as uncompressed
			// (1 block size must equal 1 uncompressed pixel size)
			EICF_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT		= 0x1u << 7u,
			//! can create with flags not supported by primary image but by a potential compatible view
			EICF_EXTENDED_USAGE_BIT						= 0x1u << 8u,
			//! irrelevant now - no support for planar images
			EICF_DISJOINT_BIT							= 0x1u << 9u,
			//! irrelevant now - two `IGPUImage`s backing memory can overlap
			EICF_ALIAS_BIT								= 0x1u << 10u,
			//! irrelevant now - we don't support protected DRM paths
			EICF_PROTECTED_BIT							= 0x1u << 11u,
			//! whether image can be used as  depth/stencil attachment with custom MSAA sample locations
			EICF_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT	= 0x1u << 12u
		};
		enum E_IMAGE_TYPE : uint32_t
		{
			EIT_1D,
			EIT_2D,
			EIT_3D
		};
		enum E_SAMPLE_COUNT_FLAGS : uint32_t
		{
			ESCF_1_BIT = 0x00000001,
			ESCF_2_BIT = 0x00000002,
			ESCF_4_BIT = 0x00000004,
			ESCF_8_BIT = 0x00000008,
			ESCF_16_BIT = 0x00000010,
			ESCF_32_BIT = 0x00000020,
			ESCF_64_BIT = 0x00000040
		};
		enum E_IMAGE_TILING : uint32_t
		{
			EIT_OPTIMAL,
			EIT_LINEAR
		};
		struct SSubresourceRange
		{
			E_IMAGE_ASPECT_FLAGS	aspectMask = static_cast<E_IMAGE_ASPECT_FLAGS>(0u); // waits for vulkan
			uint32_t				baseMipLevel = 0u;
			uint32_t				levelCount = 0u;
			uint32_t				baseArrayLayer = 0u;
			uint32_t				layerCount = 0u;
		};
		struct SSubresourceLayers
		{
			E_IMAGE_ASPECT_FLAGS	aspectMask = static_cast<E_IMAGE_ASPECT_FLAGS>(0u); // waits for vulkan
			uint32_t				mipLevel = 0u;
			uint32_t				baseArrayLayer = 0u;
			uint32_t				layerCount = 0u;
		};
		struct SBufferCopy
		{
			inline const auto&			getDstSubresource() const {return imageSubresource;}
			inline const VkOffset3D&	getDstOffset() const { return imageOffset; }
			inline const VkExtent3D&	getExtent() const { return imageExtent; }

			size_t				bufferOffset = 0ull;
			// setting this to different from 0 can fail an image copy on OpenGL
			uint32_t			bufferRowLength = 0u;
			// setting this to different from 0 can fail an image copy on OpenGL
			uint32_t			bufferImageHeight = 0u;
			SSubresourceLayers	imageSubresource;
			VkOffset3D			imageOffset = {0u,0u,0u};
			VkExtent3D			imageExtent = {0u,0u,0u};
		};
		struct SImageCopy
		{
			inline const auto&			getDstSubresource() const {return dstSubresource;}
			inline const VkOffset3D&	getDstOffset() const { return dstOffset; }
			inline const VkExtent3D&	getExtent() const { return extent; }

			SSubresourceLayers	srcSubresource;
			VkOffset3D			srcOffset = {0u,0u,0u};
			SSubresourceLayers	dstSubresource;
			VkOffset3D			dstOffset = {0u,0u,0u};
			VkExtent3D			extent = {0u,0u,0u};
		};
/*
		//!
		template<typename CopyStructIt>
		static bool validateMipchain(CopyStructIt pRegionsStart, CopyStructIt pRegionsEnd, E_IMAGE_TYPE type)
		{
			if (pRegionsStart==pRegionsEnd)
				return false;

			for (auto it = pRegionsStart; it != pRegionsEnd; it++)
			{
				constexpr uint32_t kMaxMipLevel = 16u;
				// check max size and array count
				VkOffset3D maxPt = {it->imageOffset.x+it->imageExtent.width,it->imageOffset.y+it->imageExtent.height,it->imageOffset.z+it->imageExtent.depth};
				if (*std::max_element(&maxPt.x, &maxPt.x + 3) > ((0x1u<<kMaxMipLevel) >> it->imageSubresource.mipLevel))
					return false;
				if (it->imageSubresource.mipLevel > kMaxMipLevel)
					return false;
				if (it->imageSubresource.baseArrayLayer+it->imageSubresource.layerCount > 8192u)
					return false;

				// dimension consistent with type
				switch (type)
				{
					case EIT_1D:
						if (maxPt.y > 1u)
							return false;
						_IRR_FALLTHROUGH;
					case EIT_2D:
						if (maxPt.z > 1u)
							return false;
						break;
					default: //	type unknown, or 3D format
						break;
				}

				// check regions don't overlap
			}

			return !allUnknownFmt;
		}
*/
		//!
		template<typename CopyStructIt>
		inline static auto calculateDstSizeArrayCountAndMipLevels(CopyStructIt pRegionsStart, CopyStructIt pRegionsEnd)
		{
			std::tuple<VkExtent3D,uint32_t,uint32_t> size_arraycnt_mips = {{0u,0u,0u},0u,0u};
			for (auto it=pRegionsStart; it!=pRegionsEnd; it++)
			{
				const auto& o = it->getDstOffset();
				const auto& e = it->getExtent();
				for (auto i=0; i<3; i++)
				{
					auto& inout = (&std::get<VkExtent3D>(size_arraycnt_mips).width)[i];
					inout = core::max(inout,(&e.width)[i]+(&o.width)[i]);
				}
				const auto& sub = it->getDstSubresource();
				std::get<1u>(size_arraycnt_mips) = core::max(std::get<1u>(size_arraycnt_mips),sub.baseArrayLayer+sub.layerCount);
				std::get<2u>(size_arraycnt_mips) = core::max(std::get<2u>(size_arraycnt_mips),sub.mipLevel);
			}
			std::get<2u>(size_arraycnt_mips)++;
			return size_arraycnt_mips;
		}


		//!
		E_CATEGORY getTypeCategory() const override { return EC_IMAGE; }


		//!
		inline E_IMAGE_CREATE_FLAGS getFlags() const { return flags; }

		//!
		inline E_IMAGE_TYPE getType() const { return type; }

		//! Returns the color format
		inline E_FORMAT getColorFormat() const { return format; }

		//! Returns bits per pixel.
		inline core::rational<uint32_t> getBytesPerPixel() const
		{
			return asset::getBytesPerPixel(getColorFormat());
		}

		//!
		inline auto getSize() const
		{
			return extent;
		}

		//!
		inline uint32_t getMipLevels() const
		{
			return mipLevels;
		}

		//!
		inline auto getArrayLayers() const
		{
			return arrayLayers;
		}

		//! Returns image data size in bytes
		inline size_t getImageDataSizeInBytes() const
		{
			const core::vector3du32_SIMD unit(1u);
			const auto blockAlignment = asset::getBlockDimensions(getColorFormat());
			const bool hasAnAlignment = (blockAlignment != unit).any();

			core::rational<size_t> bytesPerPixel = getBytesPerPixel();
			auto _size = core::vector3du32_SIMD(extent.width, extent.height, extent.depth);
			size_t memreq = 0ul;
			for (uint32_t i=0u; i<mipLevels; i++)
			{
				auto levelSize = _size;
				// alignup (but with NPoT alignment)
				if (hasAnAlignment)
				{
					levelSize += blockAlignment - unit;
					levelSize /= blockAlignment;
					levelSize *= blockAlignment;
				}
				auto memsize = size_t(levelSize[0] * levelSize[1])*size_t(levelSize[2] * arrayLayers)*bytesPerPixel;
				assert(memsize.getNumerator() % memsize.getDenominator() == 0u);
				memreq += memsize.getIntegerApprox();
				_size = _size / 2u;
			}
			return memreq;
		}

		//!
		inline auto getSampleCount() const
		{
			return samples;
		}

    protected:
		IImage() : flags(static_cast<E_IMAGE_CREATE_FLAGS>(0u)), type(EIT_2D), format(EF_R8G8B8A8_SRGB),
			extent({0u,0u,0u}), mipLevels(0u), arrayLayers(0u), samples(static_cast<E_SAMPLE_COUNT_FLAGS>(0u))/*,
			tiling(_tiling), usage(_usage), sharingMode(_sharingMode),
			queueFamilyIndices(_queueFamilyIndices), initialLayout(_initialLayout)*/
		{
		}
		IImage(	E_IMAGE_CREATE_FLAGS _flags,
				E_IMAGE_TYPE _type,
				E_FORMAT _format,
				const VkExtent3D& _extent,
				uint32_t _mipLevels,
				uint32_t _arrayLayers,
				E_SAMPLE_COUNT_FLAGS _samples/*,
				E_IMAGE_TILING _tiling,
				E_IMAGE_USAGE_FLAGS _usage,
				E_SHARING_MODE _sharingMode,
				core::smart_refctd_dynamic_aray<uint32_t>&& _queueFamilyIndices,
				E_IMAGE_LAYOUT _initialLayout*/)
					: flags(_flags), type(_type), format(_format), extent(_extent),
					mipLevels(_mipLevels), arrayLayers(_arrayLayers), samples(_samples)/*,
					tiling(_tiling), usage(_usage), sharingMode(_sharingMode),
					queueFamilyIndices(_queueFamilyIndices), initialLayout(_initialLayout)*/
        {
        }

		virtual ~IImage()
		{}

		E_IMAGE_CREATE_FLAGS						flags;
		E_IMAGE_TYPE								type;
		E_FORMAT									format;
		VkExtent3D									extent;
		uint32_t									mipLevels;
		uint32_t									arrayLayers;
		E_SAMPLE_COUNT_FLAGS						samples;
		//E_IMAGE_TILING							tiling;
		//E_IMAGE_USAGE_FLAGS						usage;
		//E_SHARING_MODE							sharingMode;
		//core::smart_refctd_dynamic_aray<uint32_t>	queueFamilyIndices;
		//E_IMAGE_LAYOUT							initialLayout;
};
static_assert(sizeof(IImage)-sizeof(IDescriptor)!=3u*sizeof(uint32_t)+sizeof(VkExtent3D)+sizeof(uint32_t)*3u,"BaW File Format won't work");

} // end namespace video
} // end namespace irr

#endif


