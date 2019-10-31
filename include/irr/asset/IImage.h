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
			inline bool					isValid() const
			{
				// TODO: more complex check of compatible aspects
				if (false)
					return false;

				return true;
			}

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
			inline bool					isValid() const
			{
				// TODO: more complex check of compatible aspects when planar format support arrives
				if (srcSubresource.aspectMask^dstSubresource.aspectMask)
					return false;

				if (srcSubresource.layerCount!=dstSubresource.layerCount)
					return false;

				return true;
			}

			inline const auto&			getDstSubresource() const {return dstSubresource;}
			inline const VkOffset3D&	getDstOffset() const { return dstOffset; }
			inline const VkExtent3D&	getExtent() const { return extent; }

			SSubresourceLayers	srcSubresource;
			VkOffset3D			srcOffset = {0u,0u,0u};
			SSubresourceLayers	dstSubresource;
			VkOffset3D			dstOffset = {0u,0u,0u};
			VkExtent3D			extent = {0u,0u,0u};
		};

		//!
		inline static uint32_t calculateMaxMipLevel(const VkExtent3D& extent, E_IMAGE_TYPE type)
		{
			uint32_t maxSideLen = extent.width;
			switch (type)
			{
				case EIT_2D:
					maxSideLen = core::max(extent.height,maxSideLen);
					break;
				case EIT_3D:
					maxSideLen = core::max(extent.depth,maxSideLen);
					break;
				default:
					break;
			}
			return 1u + uint32_t(floorf(log2(float(maxSideLen))));
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
		inline static bool validateCreationParameters(
			E_IMAGE_CREATE_FLAGS _flags,
			E_IMAGE_TYPE _type,
			E_FORMAT _format,
			const VkExtent3D& _extent,
			uint32_t _mipLevels,
			uint32_t _arrayLayers,
			E_SAMPLE_COUNT_FLAGS _samples)
		{

			if (_extent.width == 0u || _extent.height == 0u || _extent.depth == 0u)
				return false;
			if (_mipLevels == 0u || _arrayLayers == 0u)
				return false;

			if (_flags & EICF_CUBE_COMPATIBLE_BIT)
			{
				if (_type != EIT_2D)
					return false;
				if (_extent.width != _extent.height)
					return false;
				if (_arrayLayers < 6u)
					return false;
				if (_samples != ESCF_1_BIT)
					return false;
			}
			if ((_flags & EICF_2D_ARRAY_COMPATIBLE_BIT) && _type != EIT_3D)
				return false;
			if ((_flags & EICF_SPARSE_RESIDENCY_BIT) || (_flags & EICF_SPARSE_ALIASED_BIT))
			{
				if (!(_flags & EICF_SPARSE_BINDING_BIT))
					return false;
				if (_flags & EICF_PROTECTED_BIT)
					return false;
			}
			if ((_flags & EICF_SPARSE_BINDING_BIT) && (_flags & EICF_PROTECTED_BIT))
				return false;
			if (_flags & EICF_SPLIT_INSTANCE_BIND_REGIONS_BIT)
			{
				if (_mipLevels > 1u || _arrayLayers > 1u || _type != EIT_2D)
					return false;
			}
			if (_flags & EICF_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT)
			{
				if (!isBlockCompressionFormat(_format) || !(_flags & EICF_MUTABLE_FORMAT_BIT))
					return false;
			}
			if ((_flags & EICF_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT) && (!isDepthOrStencilFormat(_format) || _format == EF_S8_UINT))
				return false;

			if (_samples != ESCF_1_BIT && _type != EIT_2D)
				return false;

			switch (_type)
			{
				case EIT_1D:
					if (_extent.height > 1u)
						return false;
					_IRR_FALLTHROUGH;
				case EIT_2D:
					if (_extent.depth > 1u)
						return false;
					break;
				default: //	3D format
					if (_arrayLayers > 1u)
						return false;
					break;
			}

			if (asset::isPlanarFormat(_format))
			{
				if (_mipLevels > 1u || _samples != ESCF_1_BIT || _type != EIT_2D)
					return false;
			}
			else
			{
				if (!(_flags & EICF_ALIAS_BIT) && (_flags & EICF_DISJOINT_BIT))
					return false;
			}

			if (_mipLevels > calculateMaxMipLevel(_extent, _type))
				return false;

			return true;
		}

		//!
		template<typename CopyStructIt>
		static bool validatePotentialCopies(CopyStructIt pRegionsBegin, CopyStructIt pRegionsEnd, const asset::IBuffer* srcBuff)
		{
			if (pRegionsBegin==pRegionsEnd)
				return false;

			auto blockSize = getBlockDimensions(format);
			for (auto it=pRegionsBegin; it!=pRegionsEnd; it++)
			{
				if (!validatePotentialCopies_shared(pRegionsEnd,it))
					return false;

				// count on the user not being an idiot
				#ifdef _IRR_DEBUG
					size_t imageHeight = it->bufferImageHeight ? it->bufferImageHeight:it->imageExtent.height;
					imageHeight += blockSize.y-1u;
					imageHeight /= blockSize.y;
					size_t rowLength = it->bufferRowLength ? it->bufferRowLength:it->imageExtent.width;
					rowLength += blockSize.x-1u;
					rowLength /= blockSize.x;

					size_t maxBufferOffset = (it->imageExtent.depth+blockSize.z-1u)/blockSize.z-1u;
					maxBufferOffset *= imageHeight;
					maxBufferOffset += (it->imageExtent.height+blockSize.y-1u)/blockSize.y-1u;
					maxBufferOffset *= rowLength;
					maxBufferOffset += (it->imageExtent.width+blockSize.x-1u)/blockSize.z-1u;
					maxBufferOffset = (maxBufferOffset+1u)*asset::getBlockByteSize(format);
					maxBufferOffset += it->bufferOffset;
					if (maxBufferOffset>srcBuff->getSize())
					{
						assert(false);
						return false;
					}
				#endif
			}

			return true;
		}

		//!
		template<typename CopyStructIt>
		static bool validatePotentialCopies(CopyStructIt pRegionsBegin, CopyStructIt pRegionsEnd, const asset::IImage* srcImage)
		{
			if (pRegionsBegin==pRegionsEnd)
				return false;

			// tackle when we handle aspects
			//if (!asset::areFormatsCompatible(format,srcImage->format))
				//return false;
			
			auto getRealDepth = [](auto _type, auto _extent, auto subresource) -> uint32_t
			{
				if (_type!=EIT_3D)
					return subresource.arrayLayers;
				if (subresource.baseArrayLevel!=0u||subresource.arrayLayers!=1u)
					return 0u;
				return _extent.depth;
			};

			for (auto it=pRegionsBegin; it!=pRegionsEnd; it++)
			{
				if (!validatePotentialCopies_shared(pRegionsEnd,it))
					return false;

				if (it->srcSubresource.aspectMask&(~src->getAspectMask()))
					return false;
				
				auto tmp = getRealDepth(type, it->extent, it->dstSubresource);
				if (tmp!=0u || tmp!=getRealDepth(src->getType(),it->extent,it->srcSubresource))
					return false;

				// count on the user not being an idiot
				#ifdef _IRR_DEBUG
					if (it->srcSubresource.mipLayer>=src->getMipLevels())
					{
						assert(false);
						return false;
					}
					if (it->srcSubresource.baseArrayLayer+it->srcSubresource.layerCount >= src->getArrayLayerCount())
					{
						assert(false);
						return false;
					}

					const auto& off = it->srcOffset();
					const auto& ext = it->extent;
					switch (src->getType())
					{
						case EIT_1D:
							if (off.y>0u || ext.height>1u)
								return false;
							_IRR_FALLTHROUGH;
						case EIT_2D:
							if (off.z>0u || ext.depth>1u)
								return false;
							break;
						default:
							break;
					}

					auto minPt = core::vector3du32_SIMD(off.x,off.y,off.z);
					auto srcBlockDims = asset::getBlockDimensions(src->getFormat());
					if (((minPt%srcBlockDims)!=zero).any())
					{
						assert(false);
						return false;
					}

					auto maxPt = core::vector3du32_SIMD(ext.width,ext.height,ext.depth)+minPt;
					auto srcMipSize = srcImage->getMipSize(it->srcSubresource.mipLayer);
					if ((maxPt>srcMipSize || maxPt!=srcMipSize && ((maxPt%sourceBlockDims)!=zero)).any())
					{
						assert(false);
						return false;
					}
				#endif
			}

			return true;
		}

		//!
		virtual bool validateCopies(const SBufferCopy* pRegionsBegin, const SBufferCopy* pRegionsEnd, const asset::ICPUBuffer* src)
		{
			return validateCopies_template(pRegionsBegin, pRegionsEnd, src);
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
		inline const auto& getSize() const
		{
			return extent;
		}
		//!
		inline core::vector3du32_SIMD getMipSize(uint32_t level=0u) const
		{
			return core::max(core::vector3du32_SIMD(extent.width, extent.height, extent.depth) / (0x1u<<level), core::vector3du32_SIMD(1u));
		}

		//!
		inline uint32_t getMipLevelCount() const
		{
			return mipLevels;
		}

		//!
		inline auto getArrayLayerCount() const
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
			size_t memreq = 0ul;
			for (uint32_t i=0u; i<mipLevels; i++)
			{
				auto levelSize = getMipSize(i);
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
		
		template<typename CopyStructIt, class SourceType>
		inline bool validateCopies_template(CopyStructIt pRegionsBegin, CopyStructIt pRegionsEnd, const SourceType* src)
		{
			//if (flags&EICF_SUBSAMPLED)
				//return false;

			if (validatePotentialCopies(pRegionsBegin, pRegionsEnd, src))
				return false;

			const core::vector3du32_SIMD zero(0);
			auto extentSIMD = core::vector3du32_SIMD(extent.width,extent.height,extent.depth);
			for (auto it=pRegionsBegin; it!=pRegionsEnd; it++)
			{
				// check rectangles countained in destination
				const auto& subresource = it->getDstSubresource();
				if ((subresource.aspectMask&(~aspectMask))/* || !formatHasAspects(format,subresource.aspectMask)*/)
					return false;
				if (subresource.mipLevel >= mipLevels)
					return false;
				if (subresource.baseArrayLayer+subresource.layerCount >= arrayLayers)
					return false;

				const auto& off2 = it->getDstOffset();
				const auto& ext2 = it->getExtent();
				switch (src->getType())
				{
					case EIT_1D:
						if (off2.y>0u||ext2.height>1u)
							return false;
						_IRR_FALLTHROUGH;
					case EIT_2D:
						if (off2.z>0u||ext2.depth>1u)
							return false;
						break;
					default:
						break;
				}

				auto minPt2 = core::vector3du32_SIMD(off2.x,off2.y,off2.z);
				auto dstBlockDims = asset::getBlockDimensions(format);
				if ((minPt2%dstBlockDims!=zero).any())
					return false;

				auto maxPt2 = core::vector3du32_SIMD(ext2.width,ext2.height,ext2.depth);
				IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_base_of<IImage,SourceType>::value)
				{
					auto srcBlockDims = asset::getBlockDimensions(src->getFormat());

					/** Vulkan 1.1 Spec: When copying between compressed and uncompressed formats the extent members
					represent the texel dimensions of the source image and not the destination. */
					maxPt2 *= dstBlockDims/sourceBlockDims;

					// TODO: The union of all source regions, and the union of all destination regions, specified by the elements of pRegions, must not overlap in memory
				}
				IRR_PSEUDO_IF_CONSTEXPR_ELSE
				{
					// TODO: The union of all source regions, and the union of all destination regions, specified by the elements of pRegions, must not overlap in memory
				}
				IRR_PSEUDO_IF_CONSTEXPR_END
				maxPt2 += minPt2;
				if ((maxPt2>extentSIMD).any())
					return false;
			}

			return;
		}

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

	private:
		template<typename CopyStructIt>
		inline static bool validatePotentialCopies_shared(CopyStructIt pRegionsEnd, CopyStructIt it)
		{
			if (!it->isValid())
				return false;

			IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_base_of<IImage,SourceType>::value)
			{
				if (samples!=src->getSamples())
					return false;
			}
			IRR_PSEUDO_ELSE_CONSTEXPR
			{
				if (samples!=ESCF_1_BIT)
					return false;
			}
			IRR_PSEUDO_IF_CONSTEXPR_END

			constexpr uint32_t kMaxArrayCount = 8192u;
			constexpr uint32_t kMaxMipLevel = 16u;
			// check max size and array count
			const auto& off = it->getDstOffset();
			const auto& ext = it->getExtent();
			const auto& subresource = it->getDstSubresource();
			core::vector3du32_SIMD minPt(off.x,off.y,off.z);
			auto maxPt = core::vector3du32_SIMD(ext.width,ext.height,ext.depth)+minPt;
			if (*std::max_element(&maxPt.pointer, &maxPt.pointer+3) > ((0x1u<<kMaxMipLevel) >> subresource.mipLevel))
				return false;
			if (subresource.baseArrayLayer+subresource.layerCount > kMaxArrayCount)
				return false;
			if (subresource.mipLevel > kMaxMipLevel)
				return false;

			// check regions don't overlap (function is complete)
			#ifdef _IRR_DEBUG
			for (auto it2=it+1u; it2!=pRegionsEnd; it2++)
			{
				const auto& subresource2 = it2->getDstSubresource();
				if (!(subresource2.aspectMask&subresource.aspectMask))
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

} // end namespace video
} // end namespace irr

#endif


