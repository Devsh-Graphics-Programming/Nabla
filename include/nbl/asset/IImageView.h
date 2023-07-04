// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_IMAGE_VIEW_H_INCLUDED_
#define _NBL_ASSET_I_IMAGE_VIEW_H_INCLUDED_

#include "nbl/asset/IImage.h"

namespace nbl
{
namespace asset
{

template<class ImageType>
class IImageView : public IDescriptor
{
	public:
		static inline constexpr uint32_t remaining_mip_levels = ~static_cast<uint32_t>(0u);
		static inline constexpr uint32_t remaining_array_layers = ~static_cast<uint32_t>(0u);

		// no flags for now, yet
		enum E_CREATE_FLAGS
		{
			ECF_NONE = 0
		};
		enum E_TYPE
		{
			ET_1D = 0,
			ET_2D,
			ET_3D,
			ET_CUBE_MAP,
			ET_1D_ARRAY,
			ET_2D_ARRAY,
			ET_CUBE_MAP_ARRAY,
			ET_COUNT
		};
		enum E_CUBE_MAP_FACE
		{
			ECMF_POSITIVE_X = 0,
			ECMF_NEGATIVE_X,
			ECMF_POSITIVE_Y,
			ECMF_NEGATIVE_Y,
			ECMF_POSITIVE_Z,
			ECMF_NEGATIVE_Z,
			ECMF_COUNT
		};
		struct SComponentMapping
		{
			enum E_SWIZZLE
			{
				ES_IDENTITY = 0u,
				ES_ZERO		= 1u,
				ES_ONE		= 2u,
				ES_R		= 3u,
				ES_G		= 4u,
				ES_B		= 5u,
				ES_A		= 6u,
				ES_COUNT
			};
			E_SWIZZLE r = ES_R;
			E_SWIZZLE g = ES_G;
			E_SWIZZLE b = ES_B;
			E_SWIZZLE a = ES_A;

			E_SWIZZLE& operator[](const uint32_t ix)
			{
				assert(ix<4u);
				return (&r)[ix];
			}

			bool operator==(const SComponentMapping& rhs) const
			{
				return r == rhs.r && g == rhs.g && b == rhs.b && a == rhs.a;
			}
			bool operator!=(const SComponentMapping& rhs) const
			{
				return !operator==(rhs);
			}
		};
		struct SCreationParams
		{
			E_CREATE_FLAGS							flags = static_cast<E_CREATE_FLAGS>(0);
			// These are the set of usages for this ImageView, they must be a subset of the usages that `image` was created with.
			// If you leave it as the default NONE we'll inherit all usages from the `image`, setting it to anything else is
			// ONLY useful when creating multiple views of an image created with EXTENDED_USAGE to use different view formats.
			// Example: Create SRGB image with usage STORAGE, and two views with formats SRGB and R32_UINT. Then the SRGB view
			// CANNOT have STORAGE usage because the format doesn't support it, but the R32_UINT can.
			core::bitflag<IImage::E_USAGE_FLAGS>	subUsages = IImage::EUF_NONE;
			core::smart_refctd_ptr<ImageType>		image;
			E_TYPE									viewType;
			E_FORMAT								format;
			SComponentMapping						components = {};
			IImage::SSubresourceRange				subresourceRange = {IImage::EAF_COLOR_BIT,0,remaining_mip_levels,0,remaining_array_layers};
		};
		//!
		inline static bool validateCreationParameters(const SCreationParams& _params)
		{
			if (!_params.image)
				return false;

			if (_params.flags)
				return false;

			const auto& imgParams = _params.image->getCreationParameters();
			/* TODO: LAter
			image must have been created with a usage value containing at least one of VK_IMAGE_USAGE_SAMPLED_BIT,
			VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
			VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, VK_IMAGE_USAGE_SHADING_RATE_IMAGE_BIT_NV, or VK_IMAGE_USAGE_FRAGMENT_DENSITY_MAP_BIT_EXT
			if (imgParams.)
				return false;
			*/

			// declared usages that are not a subset
			if (!imgParams.usage.hasFlags(_params.subUsages))
				return false;

			const bool mutableFormat = imgParams.flags.hasFlags(IImage::ECF_MUTABLE_FORMAT_BIT);
			const bool blockTexelViewCompatible = imgParams.flags.hasFlags(IImage::ECF_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT);
			const auto& subresourceRange = _params.subresourceRange;
			if (mutableFormat)
			{
				// https://registry.khronos.org/vulkan/specs/1.2/html/vkspec.html#VUID-VkImageCreateInfo-flags-01573
				// BlockTexelViewCompatible implies MutableFormat
				if (blockTexelViewCompatible)
				{
					// https://registry.khronos.org/vulkan/specs/1.2/html/vkspec.html#VUID-VkImageViewCreateInfo-image-01583
					// If image was created with the VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT flag, format must be compatible with,
					// or must be an uncompressed format that is size-compatible with, the format used to create image
					if (getTexelOrBlockBytesize(_params.format)!=getTexelOrBlockBytesize(imgParams.format))
						return false;
					// https://registry.khronos.org/vulkan/specs/1.2/html/vkspec.html#VUID-VkImageViewCreateInfo-image-07072
					// If image was created with the VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT flag and
					// format is a non-compressed format, the levelCount and layerCount members of subresourceRange must both be 1
					if (!asset::isBlockCompressionFormat(_params.format) && (subresourceRange.levelCount!=1u || subresourceRange.layerCount!=1u))
						return false;
				}
				// https://registry.khronos.org/vulkan/specs/1.2/html/vkspec.html#VUID-VkImageViewCreateInfo-image-01761
				// If image was created with the VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT flag, but without the VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT flag,
				// and if the format of the image is not a multi-planar format, format must be compatible with the format used to create image
				else if (getFormatClass(_params.format)!=getFormatClass(imgParams.format))
					return false;
				else if (asset::isBlockCompressionFormat(_params.format)!=asset::isBlockCompressionFormat(imgParams.format))
					return false;

				/*
				TODO: if the format of the image is a multi-planar format, and if subresourceRange.aspectMask
				is one of VK_IMAGE_ASPECT_PLANE_0_BIT, VK_IMAGE_ASPECT_PLANE_1_BIT, or VK_IMAGE_ASPECT_PLANE_2_BIT,
				then format must be compatible with the VkFormat for the plane of the image format indicated by subresourceRange.aspectMask,
				as defined in Compatible formats of planes of multi-planar formats
				*/
			}
			else if (_params.format!=imgParams.format)
			{
				// TODO: multi-planar exceptions
				return false;
			}
			
			//! sanity checks
			
			// we have some layers
			if (subresourceRange.layerCount==0u)
				return false;

			// mip level ranges
			if (subresourceRange.baseMipLevel>=imgParams.mipLevels)
				return false;
			if (subresourceRange.levelCount!=remaining_mip_levels &&
					(subresourceRange.levelCount==0u ||
					subresourceRange.baseMipLevel+subresourceRange.levelCount>imgParams.mipLevels))
				return false;

			auto mipExtent = _params.image->getMipSize(subresourceRange.baseMipLevel);
			auto actualLayerCount = subresourceRange.layerCount;

			// the fact that source is 3D is implied by IImage::validateCreationParams
			const bool sourceIs2DCompat = imgParams.flags.hasFlags(IImage::ECF_2D_ARRAY_COMPATIBLE_BIT);
			if (subresourceRange.layerCount==remaining_array_layers)
			{
				if (sourceIs2DCompat && _params.viewType!=ET_3D)
					actualLayerCount = mipExtent.z;
				else
					actualLayerCount = imgParams.arrayLayers;
				actualLayerCount -= subresourceRange.baseArrayLayer;
			}

			// If image was created with the VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT flag, ... or must be an uncompressed format
			if (blockTexelViewCompatible && !asset::isBlockCompressionFormat(_params.format))
			{
				// In this case, the resulting image view�s texel dimensions equal the dimensions of the selected mip level divided by the compressed texel block size and rounded up.
				mipExtent = _params.image->getTexelBlockInfo().convertTexelsToBlocks(mipExtent);
				if (subresourceRange.layerCount==remaining_array_layers)
					actualLayerCount = 1;
			}
			const auto endLayer = actualLayerCount+subresourceRange.baseArrayLayer;

			bool checkLayers = true;
			switch (_params.viewType)
			{
				case ET_1D:
					if (imgParams.type!=IImage::ET_1D)
						return false;
					if (actualLayerCount>1u)
						return false;
					[[fallthrough]];
				case ET_1D_ARRAY:
					break;
				case ET_2D:
					if (imgParams.type==IImage::ET_1D)
						return false;
					if (actualLayerCount>1u)
						return false;
					[[fallthrough]];
				case ET_2D_ARRAY:
					if (imgParams.type==IImage::ET_3D)
					{
						if (!sourceIs2DCompat)
							return false;
						checkLayers = false; // has compatible flag

						if (imgParams.flags.value&(IImage::ECF_SPARSE_BINDING_BIT|IImage::ECF_SPARSE_RESIDENCY_BIT|IImage::ECF_SPARSE_ALIASED_BIT))
							return false;

						if (subresourceRange.levelCount>1u)
							return false;
						if (subresourceRange.baseArrayLayer>=mipExtent.z)
							return false;
						if (endLayer>mipExtent.z)
							return false;
					}
					break;
				case ET_CUBE_MAP:
					// https://registry.khronos.org/vulkan/specs/1.2/html/vkspec.html#VUID-VkImageViewCreateInfo-viewType-02960
					// https://registry.khronos.org/vulkan/specs/1.2/html/vkspec.html#VUID-VkImageViewCreateInfo-viewType-02962
					if (actualLayerCount!=6u)
						return false;
					[[fallthrough]];
				case ET_CUBE_MAP_ARRAY:
					// https://registry.khronos.org/vulkan/specs/1.2/html/vkspec.html#VUID-VkImageViewCreateInfo-viewType-02961
					// https://registry.khronos.org/vulkan/specs/1.2/html/vkspec.html#VUID-VkImageViewCreateInfo-viewType-02963
					if (actualLayerCount%6u)
						return false;
					if (!imgParams.flags.hasFlags(IImage::ECF_CUBE_COMPATIBLE_BIT))
						return false;
					break;
				case ET_3D:
					// checkLayers will chack the appropriate stuff
					break;
				default:
					return false;
					break;
			}

			if (checkLayers)
			{
				if (subresourceRange.baseArrayLayer>=imgParams.arrayLayers)
					return false;
				if (endLayer>imgParams.arrayLayers)
					return false;
			}
			return true;
		}

		//!
		E_CATEGORY	getTypeCategory() const override { return EC_IMAGE; }


		//!
		const SCreationParams&	getCreationParameters() const { return params; }

	protected:
		IImageView() : params{static_cast<E_CREATE_FLAGS>(0u),nullptr,ET_COUNT,EF_UNKNOWN,{}} {}
		IImageView(SCreationParams&& _params) : params(_params) {}
		virtual ~IImageView() = default;

		SCreationParams params;
};

}
}

#endif