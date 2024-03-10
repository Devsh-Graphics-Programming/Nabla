// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_IMAGE_H_INCLUDED_
#define _NBL_ASSET_I_IMAGE_H_INCLUDED_

#include "nbl/core/util/bitflag.h"
#include "nbl/core/containers/refctd_dynamic_array.h"
#include "nbl/core/math/glslFunctions.tcc"

#include "nbl/asset/format/EFormat.h"
#include "nbl/asset/IBuffer.h"
#include "nbl/asset/IDescriptor.h"
#include "nbl/asset/ICPUBuffer.h"
#include "nbl/asset/ECommonEnums.h"
#include "nbl/system/ILogger.h"

#include <bitset>
#include <compare>

namespace nbl::asset
{

// TODO: Vulkan's VkOffset3D has int32_t members, getting rid of this
// produces a bunch of errors in the filtering APIs and core::vectorSIMD**.
// When we have our own HLSL lib, replace these types with `uvec3`. 

//placeholder until we configure Vulkan SDK
typedef struct VkOffset3D {
	uint32_t	x;
	uint32_t	y;
	uint32_t	z;
} VkOffset3D; //depr
inline bool operator!=(const VkOffset3D& v1, const VkOffset3D& v2)
{
	return v1.x!=v2.x||v1.y!=v2.y||v1.z!=v2.z;
}
inline bool operator==(const VkOffset3D& v1, const VkOffset3D& v2)
{
	return !(v1 != v2);
}

typedef struct VkExtent3D {
	uint32_t	width;
	uint32_t	height;
	uint32_t	depth;
} VkExtent3D; //depr

inline bool operator!=(const VkExtent3D& v1, const VkExtent3D& v2)
{
	return v1.width!=v2.width||v1.height!=v2.height||v1.depth!=v2.depth;
}
inline bool operator==(const VkExtent3D& v1, const VkExtent3D& v2)
{
	return !(v1 != v2);
}


class IImage : public IDescriptor
{
	public:
		enum E_ASPECT_FLAGS : uint16_t
		{
			EAF_NONE				= 0u,
			EAF_COLOR_BIT			= 0x1u << 0u,
			EAF_DEPTH_BIT			= 0x1u << 1u,
			EAF_STENCIL_BIT			= 0x1u << 2u,
			EAF_METADATA_BIT		= 0x1u << 3u,
			EAF_PLANE_0_BIT			= 0x1u << 4u,
			EAF_PLANE_1_BIT			= 0x1u << 5u,
			EAF_PLANE_2_BIT			= 0x1u << 6u,
			EAF_MEMORY_PLANE_0_BIT	= 0x1u << 7u,
			EAF_MEMORY_PLANE_1_BIT	= 0x1u << 8u,
			EAF_MEMORY_PLANE_2_BIT	= 0x1u << 9u,
			EAF_MEMORY_PLANE_3_BIT	= 0x1u << 10u
		};
		enum E_CREATE_FLAGS : uint16_t
		{
			ECF_NONE									= 0x0u,
			//! irrelevant now - no support for sparse or aliased resources
			ECF_SPARSE_BINDING_BIT						= 0x1u << 0u,
			ECF_SPARSE_RESIDENCY_BIT					= 0x1u << 1u,
			ECF_SPARSE_ALIASED_BIT						= 0x1u << 2u,
			//! if you want to be able to create an ImageView with a different format later
			ECF_MUTABLE_FORMAT_BIT						= 0x1u << 3u, // TODO: deduce this anyway
			//! whether can fashion a cubemap out of the image
			ECF_CUBE_COMPATIBLE_BIT						= 0x1u << 4u,
			//! whether can fashion a 2d array texture out of the image
			ECF_2D_ARRAY_COMPATIBLE_BIT					= 0x1u << 5u,
			//! irrelevant now - we don't support device groups
			ECF_SPLIT_INSTANCE_BIND_REGIONS_BIT			= 0x1u << 6u,
			//! whether can view a block compressed texture as uncompressed
			// (1 block size must equal 1 uncompressed pixel size)
			ECF_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT			= 0x1u << 7u, // TODO: deduce this anyway
			//! can create with flags not supported by primary image but by a potential compatible view
			ECF_EXTENDED_USAGE_BIT						= 0x1u << 8u, // TODO: deduce this anyway
			//! irrelevant now - no support for planar images
			ECF_DISJOINT_BIT							= 0x1u << 9u,
			//! irrelevant now - two `IGPUImage`s backing memory can overlap
			ECF_ALIAS_BIT								= 0x1u << 10u,
			//! irrelevant now - we don't support protected DRM paths
			ECF_PROTECTED_BIT							= 0x1u << 11u,
			//! whether image can be used as  depth/stencil attachment with custom MSAA sample locations
			ECF_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT	= 0x1u << 12u
		};
		enum E_TYPE : uint8_t
		{
			ET_1D = 0,
			ET_2D,
			ET_3D,
			ET_COUNT
		};
		enum E_SAMPLE_COUNT_FLAGS : uint8_t
		{
			ESCF_1_BIT = 0x01,
			ESCF_2_BIT = 0x02,
			ESCF_4_BIT = 0x04,
			ESCF_8_BIT = 0x08,
			ESCF_16_BIT = 0x10,
			ESCF_32_BIT = 0x20,
			ESCF_64_BIT = 0x40
		};
		enum E_USAGE_FLAGS : uint16_t
		{
			EUF_NONE = 0x0000,
			EUF_TRANSFER_SRC_BIT = 0x0001,
			EUF_TRANSFER_DST_BIT = 0x0002,
			EUF_SAMPLED_BIT = 0x0004,
			EUF_STORAGE_BIT = 0x0008,
			EUF_RENDER_ATTACHMENT_BIT = 0x0010,
			EUF_TRANSIENT_ATTACHMENT_BIT = 0x0040,
			EUF_INPUT_ATTACHMENT_BIT = 0x0080,
			EUF_SHADING_RATE_ATTACHMENT_BIT = 0x0100,
			EUF_FRAGMENT_DENSITY_MAP_BIT = 0x0200
		};
		struct SSubresourceRange
		{
			core::bitflag<E_ASPECT_FLAGS>	aspectMask = E_ASPECT_FLAGS::EAF_NONE;
			uint32_t						baseMipLevel = 0u;
			uint32_t						levelCount = 0u;
			uint32_t						baseArrayLayer = 0u;
			uint32_t						layerCount = 0u;
		};
		struct SSubresourceLayers
		{
			core::bitflag<E_ASPECT_FLAGS>	aspectMask = E_ASPECT_FLAGS::EAF_NONE;
			uint32_t						mipLevel = 0u;
			uint32_t						baseArrayLayer = 0u;
			uint32_t						layerCount = 0u;

			auto operator<=>(const SSubresourceLayers&) const = default;
		};
		struct SBufferCopy
		{
			inline bool					isValid() const
			{
				// TODO: more complex check of compatible aspects
				// Image Extent must be a mutiple of texel block dims OR offset + extent = image subresourceDims
				// bufferOffset must be multiple of the compressed texel block size in bytes (matters in IGPU?)
				// If planar subresource aspectMask should be PLANE_{0,1,2}
				if (false)
					return false;

				return true;
			}

			inline const auto&			getDstSubresource() const {return imageSubresource;}
			inline const VkOffset3D&	getDstOffset() const { return imageOffset; }
			inline const VkExtent3D&	getExtent() const { return imageExtent; }


			inline auto					getTexelStrides() const
			{
				core::vector3du32_SIMD trueExtent;
				trueExtent.x = bufferRowLength ? bufferRowLength:imageExtent.width;
				trueExtent.y = bufferImageHeight ? bufferImageHeight:imageExtent.height;
				trueExtent.z = imageExtent.depth;
				return trueExtent;
			}

			inline auto					getBlockStrides(const TexelBlockInfo& info) const
			{
				return info.convertTexelsToBlocks(getTexelStrides());
			}


			inline auto					getByteStrides(const TexelBlockInfo& info) const
			{
				return info.convert3DTexelStridesTo1DByteStrides(getTexelStrides());
			}
			static inline uint64_t		getLocalByteOffset(const core::vector3du32_SIMD& localXYZLayerOffset, const core::vector3du32_SIMD& byteStrides)
			{
				return core::dot(localXYZLayerOffset,byteStrides)[0];
			}
			inline uint64_t				getByteOffset(const core::vector3du32_SIMD& localXYZLayerOffset, const core::vector3du32_SIMD& byteStrides) const
			{
				return bufferOffset+getLocalByteOffset(localXYZLayerOffset,byteStrides);
			}

			size_t				bufferOffset = 0ull;
			// setting this to different from 0 can fail an image copy on OpenGL
			uint32_t			bufferRowLength = 0u;
			// setting this to different from 0 can fail an image copy on OpenGL
			uint32_t			bufferImageHeight = 0u; // TODO: size_t ?
			SSubresourceLayers	imageSubresource = {};
			VkOffset3D			imageOffset = {0u,0u,0u};
			VkExtent3D			imageExtent = {0u,0u,0u};

			auto operator<=>(const SBufferCopy&) const = default;
		};
		struct SImageCopy
		{
			inline bool					isValid() const
			{
				// TODO: more complex check of compatible aspects when planar format support arrives
				if ((srcSubresource.aspectMask^dstSubresource.aspectMask).value)
					return false;

				if (srcSubresource.layerCount!=dstSubresource.layerCount)
					return false;

				return true;
			}

			inline const auto&			getDstSubresource() const {return dstSubresource;}
			inline const VkOffset3D&	getDstOffset() const { return dstOffset; }
			inline const VkExtent3D&	getExtent() const { return extent; }

			SSubresourceLayers	srcSubresource = {};
			VkOffset3D			srcOffset = {0u,0u,0u};
			SSubresourceLayers	dstSubresource = {};
			VkOffset3D			dstOffset = {0u,0u,0u};
			VkExtent3D			extent = {0u,0u,0u};
		};
		struct SCreationParams
		{
			E_TYPE							type;
			E_SAMPLE_COUNT_FLAGS			samples;
			E_FORMAT						format;
			VkExtent3D						extent;
			uint32_t						mipLevels;
			uint32_t						arrayLayers;
			core::bitflag<E_CREATE_FLAGS>	flags = ECF_NONE;
			union
			{
				// need to give it a nice default so validation doesn't fail when we don't touch it
				core::bitflag<E_USAGE_FLAGS>	usage = EUF_SAMPLED_BIT;
				core::bitflag<E_USAGE_FLAGS>	depthUsage;
			};
			// Do not touch AT ALL for an image without stencil format & aspect
			// Leave as is to default to `depthUsage` for a stencil image
			core::bitflag<E_USAGE_FLAGS>	stencilUsage = EUF_NONE;
			// Do not touch unless you want to fail at creating views with their Format not listed here
			std::bitset<E_FORMAT::EF_COUNT>	viewFormats = {};

			inline core::bitflag<E_USAGE_FLAGS> actualStencilUsage() const
			{
				return stencilUsage.value!=E_USAGE_FLAGS::EUF_NONE ? stencilUsage:depthUsage;
			}

			inline bool operator==(const SCreationParams& rhs) const
			{
				return !operator!=(rhs);
			}
			inline bool operator!=(const SCreationParams& rhs) const
			{
				return type!=rhs.type ||
					samples!=rhs.samples ||
					format!=rhs.format ||
					extent!=rhs.extent ||
					mipLevels!=rhs.mipLevels ||
					arrayLayers!=rhs.arrayLayers ||
					flags.value!=rhs.flags.value ||
					usage.value!=rhs.usage.value;
			}
		};

		//!
		inline const auto& getCreationParameters() const
		{
			return m_creationParams;
		}

		//!
		inline static uint32_t calculateMaxMipLevel(const VkExtent3D& extent, E_TYPE type)
		{
			uint32_t maxSideLen = extent.width;
			switch (type)
			{
				case ET_3D:
					maxSideLen = core::max<uint32_t>(extent.depth,maxSideLen);
					[[fallthrough]];
				case ET_2D:
					maxSideLen = core::max<uint32_t>(extent.height,maxSideLen);
					break;
				default:
					break;
			}
			const uint32_t round = core::roundUpToPoT<uint32_t>(maxSideLen);
			return hlsl::findLSB(round);
		}
		inline static uint32_t calculateFullMipPyramidLevelCount(const VkExtent3D& extent, E_TYPE type)
		{
			return calculateMaxMipLevel(extent,type)+1u;
		}

		//!
		template<typename CopyStructIt>
		inline static auto calculateDstSizeArrayCountAndMipLevels(CopyStructIt pRegionsStart, CopyStructIt pRegionsEnd)
		{
			std::tuple<VkExtent3D, uint32_t, uint32_t> size_arraycnt_mips = { {0u,0u,0u},0u,0u };
			for (auto it = pRegionsStart; it != pRegionsEnd; it++)
			{
				const auto& o = it->getDstOffset();
				const auto& e = it->getExtent();
				for (auto i = 0; i < 3; i++)
				{
					auto& inout = (&std::get<VkExtent3D>(size_arraycnt_mips).width)[i];
					inout = core::max(inout, (&e.width)[i] + (&o.width)[i]);
				}
				const auto& sub = it->getDstSubresource();
				std::get<1u>(size_arraycnt_mips) = core::max(std::get<1u>(size_arraycnt_mips), sub.baseArrayLayer + sub.layerCount);
				std::get<2u>(size_arraycnt_mips) = core::max(std::get<2u>(size_arraycnt_mips), sub.mipLevel);
			}
			std::get<2u>(size_arraycnt_mips)++;
			return size_arraycnt_mips;
		}

		//!
		inline static bool validateCreationParameters(const SCreationParams& _params)
		{
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-extent-00944
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-extent-00945
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-extent-00946
			if (_params.extent.width == 0u || _params.extent.height == 0u || _params.extent.depth == 0u)
				return false;

			// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-extent-00947
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-extent-00948
			if (_params.mipLevels == 0u || _params.arrayLayers == 0u)
				return false;

			if (hlsl::bitCount(static_cast<uint32_t>(_params.samples))!=1u)
				return false;

			if (_params.flags.hasFlags(ECF_CUBE_COMPATIBLE_BIT))
			{
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-flags-00949
				if (_params.type != ET_2D)
					return false;
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-flags-00950
				if (_params.extent.width != _params.extent.height)
					return false;
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-flags-08866
				if (_params.arrayLayers < 6u)
					return false;
				if (_params.samples != ESCF_1_BIT)
					return false;
			}

			// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-flags-00950
			if (_params.flags.hasFlags(ECF_2D_ARRAY_COMPATIBLE_BIT) && _params.type != ET_3D)
				return false;
			// TODO: add 2D view compat bit?
			
			const bool sparseResidency = _params.flags.hasFlags(ECF_SPARSE_RESIDENCY_BIT);
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-imageType-00970
			if (sparseResidency && _params.type == ET_1D)
				return false;

			if (sparseResidency || _params.flags.hasFlags(ECF_SPARSE_ALIASED_BIT))
			{
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-flags-00987
				if (!_params.flags.hasFlags(ECF_SPARSE_BINDING_BIT))
					return false;
				if (_params.flags.hasFlags(ECF_PROTECTED_BIT))
					return false;
			}
			if (_params.flags.hasFlags(ECF_SPARSE_BINDING_BIT) && _params.flags.hasFlags(ECF_PROTECTED_BIT))
				return false;
			if (_params.flags.hasFlags(ECF_SPLIT_INSTANCE_BIND_REGIONS_BIT))
			{
				if (_params.mipLevels > 1u || _params.arrayLayers > 1u || _params.type != ET_2D)
					return false;
			}

			const bool isMutable = _params.flags.hasFlags(ECF_MUTABLE_FORMAT_BIT);
			{
				const bool isBlockTexelViewCompat = _params.flags.hasFlags(ECF_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT);
				if (isBlockTexelViewCompat)
				{
					// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-flags-01572
					// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-flags-01573
					if (!isBlockCompressionFormat(_params.format) || !isMutable)
						return false;
				}
				bool actuallyMutable = false;
				// if only there existed a nice iterator that would let me iterate over set bits 64 faster
				if (_params.viewFormats.any())
				for (auto fmt=0; fmt<_params.viewFormats.size(); fmt++)
				if (_params.viewFormats.test(fmt))
				{
					const auto format = static_cast<asset::E_FORMAT>(fmt);
					// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-pNext-06722
					if (asset::getFormatClass(format) != asset::getFormatClass(_params.format))
					{
						if (!isBlockTexelViewCompat || asset::getTexelOrBlockBytesize(format)!=asset::getTexelOrBlockBytesize(_params.format))
							return false;
					}
					actuallyMutable = format!=_params.format;
				}
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-flags-04738
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-flags-04738
				if (actuallyMutable && !isMutable)
					return false;
			}

			const bool depthOrStencilFormat = isDepthOrStencilFormat(_params.format);
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-flags-01533
			if (_params.flags.hasFlags(ECF_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT) && (!depthOrStencilFormat || _params.format == EF_S8_UINT))
				return false;

			if (depthOrStencilFormat)
			{
				// IGNORE
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-format-02795
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-format-02796
				// WHY? Because we don't expose "combined" usage enums at all!

				// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-format-02797
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-format-02798
				if ((_params.depthUsage&EUF_TRANSIENT_ATTACHMENT_BIT).value!=(_params.stencilUsage&EUF_TRANSIENT_ATTACHMENT_BIT).value)
					return false;
			}

			if (_params.samples != ESCF_1_BIT)
			{
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-samples-02257
				if (_params.type != ET_2D || _params.flags.hasFlags(ECF_CUBE_COMPATIBLE_BIT) || _params.mipLevels == 1u)
					return false;
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-samples-02558
				if (_params.usage.hasFlags(EUF_FRAGMENT_DENSITY_MAP_BIT))
					return false;
			}

			if (_params.usage.hasFlags(EUF_TRANSIENT_ATTACHMENT_BIT))
			{
				const auto attachmentUsageFlags = core::bitflag(EUF_RENDER_ATTACHMENT_BIT)|EUF_INPUT_ATTACHMENT_BIT;
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-usage-00966
				if (!_params.usage.hasFlags(attachmentUsageFlags))
					return false;
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-usage-00963
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-None-01925
				if (_params.usage.hasFlags(~(attachmentUsageFlags|EUF_TRANSIENT_ATTACHMENT_BIT)))
					return false;
			}

			switch (_params.type)
			{
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-imageType-00956
				case ET_1D:
					if (_params.extent.height > 1u)
						return false;
					[[fallthrough]];
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-imageType-00957
				case ET_2D:
					if (_params.extent.depth > 1u)
						return false;
					break;
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-imageType-00961
				default: //	3D format
					if (_params.arrayLayers > 1u)
						return false;
					break;
			}

			if (asset::isPlanarFormat(_params.format))
			{
				if (_params.mipLevels > 1u || _params.samples != ESCF_1_BIT || _params.type != ET_2D)
					return false;
			}
			else if (!_params.flags.hasFlags(ECF_ALIAS_BIT) && _params.flags.hasFlags(ECF_DISJOINT_BIT))
				return false;

			// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-mipLevels-00958
			if (_params.mipLevels > calculateFullMipPyramidLevelCount(_params.extent, _params.type))
				return false;

			if (_params.usage.hasFlags(EUF_SHADING_RATE_ATTACHMENT_BIT))
			{
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#VUID-VkImageCreateInfo-imageType-02082
				if (_params.type!=ET_2D || _params.samples!=ESCF_1_BIT)
					return false;
			}

			return true;
		}

		//!
		template<typename CopyStructIt>
		static bool validatePotentialCopies(CopyStructIt pRegionsBegin, CopyStructIt pRegionsEnd, const asset::IBuffer* srcBuff)
		{
			if (pRegionsBegin==pRegionsEnd)
				return false;

			for (auto it=pRegionsBegin; it!=pRegionsEnd; it++)
			{
				if (!validatePotentialCopies_shared(pRegionsEnd,it))
					return false;
			}

			return true;
		}

		//!
		template<typename CopyStructIt>
		static bool validatePotentialCopies(CopyStructIt pRegionsBegin, CopyStructIt pRegionsEnd, const asset::IImage* srcImage)
		{
			if (pRegionsBegin==pRegionsEnd)
				return false;
			
			const core::vector3du32_SIMD zero(0u,0u,0u,0u);
			for (auto it=pRegionsBegin; it!=pRegionsEnd; it++)
			{
				if (!validatePotentialCopies_shared(pRegionsEnd,it))
					return false;

				// count on the user not being an idiot
#ifdef _NBL_DEBUG
				if (it->srcSubresource.mipLevel>=srcImage->getCreationParameters().mipLevels)
				{
					assert(false);
					return false;
				}
				if (it->srcSubresource.baseArrayLayer+it->srcSubresource.layerCount > srcImage->getCreationParameters().arrayLayers)
				{
					assert(false);
					return false;
				}

				const auto& off = it->srcOffset;
				const auto& ext = it->extent;
				switch (srcImage->getCreationParameters().type)
				{
					case ET_1D:
						if (off.y>0u || ext.height>1u)
							return false;
						[[fallthrough]];
					case ET_2D:
						if (off.z>0u || ext.depth>1u)
							return false;
						break;
					default:
						break;
				}

				auto minPt = core::vector3du32_SIMD(off.x,off.y,off.z);
				auto srcBlockDims = asset::getBlockDimensions(srcImage->getCreationParameters().format);
				if (((minPt%srcBlockDims)!=zero).any())
				{
					assert(false);
					return false;
				}

				auto maxPt = core::vector3du32_SIMD(ext.width,ext.height,ext.depth)+minPt;
				auto srcMipSize = srcImage->getMipSize(it->srcSubresource.mipLevel);
				if ((maxPt>srcMipSize || (maxPt!=srcMipSize && ((maxPt%srcBlockDims)!=zero))).any())
				{
					assert(false);
					return false;
				}
#endif
			}

			return true;
		}

		//!
		virtual bool validateCopies(const SBufferCopy* pRegionsBegin, const SBufferCopy* pRegionsEnd, const asset::ICPUBuffer* src) const
		{
			return validateCopies_template(pRegionsBegin, pRegionsEnd, src);
		}


		//!
		E_CATEGORY getTypeCategory() const override { return EC_IMAGE; }


		inline const auto& getTexelBlockInfo() const
		{
			return info;
		}

		//! Returns bits per pixel.
		inline core::rational<uint32_t> getBytesPerPixel() const
		{
			return asset::getBytesPerPixel(m_creationParams.format);
		}

		//!
		inline core::vector3du32_SIMD getMipSize(uint32_t level=0u) const
		{
			return core::max<core::vector3du32_SIMD>(
				core::vector3du32_SIMD(
					m_creationParams.extent.width,
					m_creationParams.extent.height,
					m_creationParams.extent.depth
				)/(0x1u<<level),
				core::vector3du32_SIMD(1u,1u,1u)
			);      
		}

		//! Returns image data size in bytes
		inline size_t getImageDataSizeInBytes() const
		{
			core::rational<size_t> bytesPerPixel = getBytesPerPixel();
			size_t memreq = 0ull;
			for (uint32_t i=0u; i<m_creationParams.mipLevels; i++)
			{
				auto levelSize = info.roundToBlockSize(getMipSize(i));
				auto memsize = size_t(levelSize[0]*levelSize[1])*size_t(levelSize[2]*m_creationParams.arrayLayers)*bytesPerPixel;
				assert(memsize.getNumerator() % memsize.getDenominator() == 0u);
				memreq += memsize.getIntegerApprox();
			}
			return memreq;
		}

		enum class LAYOUT : uint8_t
		{
			UNDEFINED,
			GENERAL,
			READ_ONLY_OPTIMAL,
			ATTACHMENT_OPTIMAL,
			TRANSFER_SRC_OPTIMAL,
			TRANSFER_DST_OPTIMAL,
			PREINITIALIZED,
			PRESENT_SRC,
			SHARED_PRESENT
			// TODO: Density Map, shading rate (are they the same thing?), and feedback loop optimal
		};
		// utility struct for later
		struct SDepthStencilLayout
		{
			LAYOUT depth = LAYOUT::UNDEFINED;
			// if you leave `stencilLayout` as undefined this means you want same layout as `depth`
			LAYOUT stencil = LAYOUT::UNDEFINED;

			auto operator<=>(const SDepthStencilLayout&) const = default;

			inline LAYOUT actualStencilLayout() const
			{
				using layout_t = LAYOUT;
				if (stencil != layout_t::UNDEFINED)
					return stencil;
				return depth;
			}
		};
	protected:
		IImage(const SCreationParams& _params) : m_creationParams(_params), info(_params.format) {}

		virtual ~IImage() {}
		
		template<typename CopyStructIt, class SourceType>
		inline bool validateCopies_template(CopyStructIt pRegionsBegin, CopyStructIt pRegionsEnd, const SourceType* src) const
		{
			//if (flags&ECF_SUBSAMPLED)
				//return false;

			if (!validatePotentialCopies(pRegionsBegin, pRegionsEnd, src))
				return false;
			
			bool die = false;
			if constexpr(std::is_base_of<IImage, SourceType>::value)
			{
				if (m_creationParams.samples!=src->getCreationParameters().samples)
					die = true;

				// tackle when we handle aspects
				//if (!asset::areFormatsCompatible(m_creationParams.format,src->format))
					//die = true;
			}
			else
			{
				if (m_creationParams.samples!=ESCF_1_BIT)
					die = true;
			}
			
			if (die)
				return false;


			const core::vector3du32_SIMD zero(0u,0u,0u,0u);
			auto extentSIMD = core::vector3du32_SIMD(m_creationParams.extent.width,m_creationParams.extent.height,m_creationParams.extent.depth);
			auto blockByteSize = asset::getTexelOrBlockBytesize(m_creationParams.format);
			auto dstBlockDims = asset::getBlockDimensions(m_creationParams.format);
			for (auto it=pRegionsBegin; it!=pRegionsEnd; it++)
			{
				// check rectangles countained in destination
				const auto& subresource = it->getDstSubresource();
				//if (!formatHasAspects(m_creationParams.format,subresource.aspectMask))
					//return false;
				// The aspectMask member of imageSubresource must only have a single bit set
				if (!hlsl::bitCount<uint32_t>(subresource.aspectMask.value) == 1u)
					return false;
				if (subresource.mipLevel >= m_creationParams.mipLevels)
					return false;
				if (subresource.baseArrayLayer+subresource.layerCount > m_creationParams.arrayLayers)
					return false;

				const auto& off2 = it->getDstOffset();
				const auto& ext2 = it->getExtent();
				switch (m_creationParams.type)
				{
					case ET_1D:
						if (off2.y>0u||ext2.height>1u)
							return false;
						[[fallthrough]];
					case ET_2D:
						if (off2.z>0u||ext2.depth>1u)
							return false;
						break;
					case ET_3D:
						if (subresource.baseArrayLayer!=0u||subresource.layerCount!=1u)
							return false;
						break;
					default:
						assert(false);
						break;
				}

				auto minPt2 = core::vector3du32_SIMD(off2.x,off2.y,off2.z);
				if ((minPt2%dstBlockDims!=zero).any())
					return false;

				auto maxPt2 = core::vector3du32_SIMD(ext2.width,ext2.height,ext2.depth);
				bool die = false;
				if constexpr(std::is_base_of<IImage,SourceType>::value)
				{
					//if (!formatHasAspects(src->m_creationParams.format,it->srcSubresource.aspectMask))
						//die = true;

					auto getRealDepth = [](auto _type, auto _extent, auto subresource) -> uint32_t
					{
						if (_type != ET_3D)
							return subresource.layerCount;
						if (subresource.baseArrayLayer != 0u || subresource.layerCount != 1u)
							return 0u;
						return _extent.depth;
					};
					if (getRealDepth(m_creationParams.type, it->extent, subresource) != getRealDepth(src->m_creationParams.type, it->extent, it->srcSubresource))
						die = true;

					auto srcBlockDims = asset::getBlockDimensions(src->m_creationParams.format);

					/** Vulkan 1.1 Spec: When copying between compressed and uncompressed formats the extent members
					represent the texel dimensions of the source image and not the destination. */
					maxPt2 *= dstBlockDims/srcBlockDims;

					// TODO: The union of all source regions, and the union of all destination regions, specified by the elements of pRegions, must not overlap in memory
				}
				else
				{
					// count on the user not being an idiot
					#ifdef _NBL_DEBUG
						size_t imageHeight = it->bufferImageHeight ? it->bufferImageHeight:it->imageExtent.height;
						imageHeight += dstBlockDims.y-1u;
						imageHeight /= dstBlockDims.y;
						size_t rowLength = it->bufferRowLength ? it->bufferRowLength:it->imageExtent.width;
						rowLength += dstBlockDims.x-1u;
						rowLength /= dstBlockDims.x;

						size_t maxBufferOffset = (it->imageExtent.depth + dstBlockDims.z - 1u) / dstBlockDims.z - 1u;
						maxBufferOffset *= imageHeight;
						maxBufferOffset += (it->imageExtent.height + dstBlockDims.y - 1u) / dstBlockDims.y - 1u;
						maxBufferOffset *= rowLength;
						maxBufferOffset += (it->imageExtent.width + dstBlockDims.x - 1u) / dstBlockDims.x - 1u;
						maxBufferOffset = (maxBufferOffset+1u)*blockByteSize;
						maxBufferOffset += it->bufferOffset;
						if (maxBufferOffset>src->getSize())
							die = true;
					#endif

					// TODO: The union of all source regions, and the union of all destination regions, specified by the elements of pRegions, must not overlap in memory
				}
				

				if (die)
					return false;

				maxPt2 += minPt2;
				if ((maxPt2>extentSIMD).any())
					return false;
			}

			return true;
		}


		SCreationParams m_creationParams;
		TexelBlockInfo  info;

	private:
		template<typename CopyStructIt>
		inline static bool validatePotentialCopies_shared(CopyStructIt pRegionsEnd, CopyStructIt it)
		{
			if (!it->isValid())
				return false;

			constexpr uint32_t kMaxArrayCount = 8192u;
			constexpr uint32_t kMaxMipLevel = 16u;
			// check max size and array count
			const auto& off = it->getDstOffset();
			const auto& ext = it->getExtent();
			const auto& subresource = it->getDstSubresource();
			core::vector3du32_SIMD minPt(off.x,off.y,off.z);
			auto maxPt = core::vector3du32_SIMD(ext.width,ext.height,ext.depth)+minPt;
			if (*std::max_element(maxPt.pointer, maxPt.pointer+3) > ((0x1u<<kMaxMipLevel) >> subresource.mipLevel))
				return false;
			if (subresource.baseArrayLayer+subresource.layerCount > kMaxArrayCount)
				return false;
			if (subresource.mipLevel > kMaxMipLevel)
				return false;

			// check regions don't overlap (function is complete)
			#ifdef _NBL_DEBUG
			for (auto it2=it+1u; it2!=pRegionsEnd; it2++)
			{
				const auto& subresource2 = it2->getDstSubresource();
				if (!(subresource2.aspectMask&subresource.aspectMask).value)
					continue;
				if (subresource2.mipLevel!=subresource.mipLevel)
					continue;
				if (subresource2.baseArrayLayer >= subresource.baseArrayLayer+subresource.layerCount)
					continue;
				if (subresource2.baseArrayLayer+subresource2.layerCount <= subresource.baseArrayLayer)
					continue;

				const auto& off2 = it2->getDstOffset();
				const auto& ext2 = it2->getExtent();
				core::vector3du32_SIMD minPt2(off2.x,off2.y,off2.z);
				auto maxPt2 = core::vector3du32_SIMD(ext2.width,ext2.height,ext2.depth)+minPt2;
				if ((minPt<maxPt2&&maxPt>minPt2).all())
					return false;
			}
			#endif

			return true;
		}
};
static_assert(sizeof(IImage)-sizeof(IDescriptor)!=3u*sizeof(uint32_t)+sizeof(VkExtent3D)+sizeof(uint32_t)*3u,"BaW File Format won't work");

NBL_ENUM_ADD_BITWISE_OPERATORS(IImage::E_USAGE_FLAGS)
NBL_ENUM_ADD_BITWISE_OPERATORS(IImage::E_SAMPLE_COUNT_FLAGS)

} // end namespace nbl::asset

#endif


