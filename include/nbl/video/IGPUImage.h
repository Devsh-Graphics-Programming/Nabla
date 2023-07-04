// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_GPU_IMAGE_H_INCLUDED__
#define __NBL_VIDEO_GPU_IMAGE_H_INCLUDED__


#include "dimension2d.h"
#include "IDeviceMemoryBacked.h"

#include "nbl/asset/IImage.h"

#include "nbl/video/IGPUBuffer.h"
#include "nbl/video/decl/IBackendObject.h"


namespace nbl::video
{

class IGPUImage : public asset::IImage, public IDeviceMemoryBacked, public IBackendObject
{
	public:
		enum E_TILING : uint8_t
		{
			ET_OPTIMAL,
			ET_LINEAR
		};
		struct SCreationParams : asset::IImage::SCreationParams, IDeviceMemoryBacked::SCreationParams
		{
			E_TILING tiling = ET_OPTIMAL;
			E_LAYOUT initialLayout = EL_UNDEFINED;

			SCreationParams& operator =(const asset::IImage::SCreationParams& rhs)
			{
				static_cast<asset::IImage::SCreationParams&>(*this) = rhs;
				return *this;
			}
		};

		//!
		inline E_TILING getTiling() const {return m_tiling;}

		//!
		inline E_LAYOUT getInitialLayout() const { return m_initialLayout; }

		//!
		E_OBJECT_TYPE getObjectType() const override { return EOT_IMAGE; }

		//!
		virtual bool validateCopies(const SBufferCopy* pRegionsBegin, const SBufferCopy* pRegionsEnd, const IGPUBuffer* src, const asset::VkExtent3D& minImageTransferGranularity = { 1,1,1 }) const
		{
			if (!validateCopies_template(pRegionsBegin, pRegionsEnd, src))
				return false;

			const auto srcParams = src->getCreationParams();
			if (!srcParams.usage.hasFlags(asset::IBuffer::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT))
				return false;

			for (auto region = pRegionsBegin; region != pRegionsEnd; region++)
			{
				auto subresourceSize = getMipSize(region->imageSubresource.mipLevel);
				if (!validateCopyOffsetAndExtent(region->imageExtent, region->imageOffset, subresourceSize, minImageTransferGranularity))
					return false;
			}

			// TODO:
			// buffer has memory bound (with sparse exceptions)
			// format features of dstImage contain transfer dst bit
			// dst image not created subsampled
			return true;
		}
		
		//!
		virtual bool validateCopies(const SImageCopy* pRegionsBegin, const SImageCopy* pRegionsEnd, const IGPUImage* src, const asset::VkExtent3D& minImageTransferGranularity = { 1,1,1 }) const
		{
			if (!validateCopies_template(pRegionsBegin, pRegionsEnd, src))
				return false;

			const auto srcParams = src->getCreationParameters();
			if (!srcParams.usage.hasFlags(asset::IImage::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT))
				return false;

			for (auto region = pRegionsBegin; region != pRegionsEnd; region++)
			{
				auto srcSubresourceSize = src->getMipSize(region->srcSubresource.mipLevel);
				if (!validateCopyOffsetAndExtent(region->extent, region->srcOffset, srcSubresourceSize, minImageTransferGranularity))
					return false;
				auto dstSubresourceSize = getMipSize(region->dstSubresource.mipLevel);
				if (!validateCopyOffsetAndExtent(region->extent, region->dstOffset, dstSubresourceSize, minImageTransferGranularity))
					return false;
			}

			// TODO:
			// buffer has memory bound (with sparse exceptions)
			// format features of dstImage contain transfer dst bit
			// dst image not created subsampled
			return true;
		}

		// ! See Vulkan Specification Notes on `VkQueueFamilyProperties::minImageTransferGranularity`
		virtual bool validateCopyOffsetAndExtent(const asset::VkExtent3D& extent, const asset::VkOffset3D& offset, const core::vector3du32_SIMD& subresourceSize, const asset::VkExtent3D& minImageTransferGranularity) const
		{
			const bool canTransferMipLevelsPartially = !(minImageTransferGranularity.width == 0 && minImageTransferGranularity.height == 0 && minImageTransferGranularity.depth == 0);
			auto texelBlockDim = asset::getBlockDimensions(m_creationParams.format);

			if (canTransferMipLevelsPartially)
			{
				// region's imageOffset.{xyz} should be multiple of minImageTransferGranularity.{xyz} scaled up by block size
				bool isImageOffsetAlignmentValid =
					(offset.x % (minImageTransferGranularity.width * texelBlockDim.x) == 0) &&
					(offset.y % (minImageTransferGranularity.height * texelBlockDim.y) == 0) &&
					(offset.z % (minImageTransferGranularity.depth * texelBlockDim.z) == 0);

				if (!isImageOffsetAlignmentValid)
					return false;

				// region's imageExtent.{xyz} should be multiple of minImageTransferGranularity.{xyz} scaled up by block size,
				// OR ELSE (offset.{x/y/z} + extent.{width/height/depth}) MUST be equal to subresource{Width,Height,Depth}
				bool isImageExtentAlignmentValid =
					(extent.width % (minImageTransferGranularity.width * texelBlockDim.x) == 0 || (offset.x + extent.width == subresourceSize.x)) &&
					(extent.height % (minImageTransferGranularity.height * texelBlockDim.y) == 0 || (offset.y + extent.height == subresourceSize.y)) &&
					(extent.depth % (minImageTransferGranularity.depth * texelBlockDim.z) == 0 || (offset.z + extent.depth == subresourceSize.z));

				if (!isImageExtentAlignmentValid)
					return false;

				bool isImageExtentAndOffsetValid =
					(extent.width + offset.x <= subresourceSize.x) &&
					(extent.height + offset.y <= subresourceSize.y) &&
					(extent.depth + offset.z <= subresourceSize.z);

				if (!isImageExtentAndOffsetValid)
					return false;
			}
			else
			{
				if (!(offset.x == 0 && offset.y == 0 && offset.z == 0))
					return false;
				if (!(extent.width == subresourceSize.x && extent.height == subresourceSize.y && extent.depth == subresourceSize.z))
					return false;
			}
			return true;
		}

		// OpenGL: const GLuint* handle of a texture target
		// Vulkan: const VkImage*
		virtual const void* getNativeHandle() const = 0;

	protected:
		const E_TILING m_tiling;
		const E_LAYOUT m_initialLayout;

		_NBL_INTERFACE_CHILD(IGPUImage) {}

		//! constructor
		IGPUImage(core::smart_refctd_ptr<const ILogicalDevice>&& dev,
			const IDeviceMemoryBacked::SDeviceMemoryRequirements& reqs,
			SCreationParams&& _params
		) : IImage(_params), IDeviceMemoryBacked(std::move(_params),reqs), IBackendObject(std::move(dev)), m_tiling(_params.tiling), m_initialLayout(_params.initialLayout) {}
};


} // end namespace nbl::video

#endif

