#ifndef _IRR_EXT_SCREEN_SHOT_INCLUDED_
#define _IRR_EXT_SCREEN_SHOT_INCLUDED_

#include "irrlicht.h"

#include "../../../../source/Irrlicht/COpenGLBuffer.h"
#include "../../../../source/Irrlicht/COpenGLExtensionHandler.h"
#include "../../../irr/asset/IImageAssetHandlerBase.h"

namespace irr
{
	namespace ext
	{
		namespace ScreenShot
		{
			/*
				Creates useful FBO that may be used for instance 
				to fetch default render target color data attachment 
				after rendering scene.
				The usage is following:
				- create frame buffer using the function
				- blit framebuffers, driver->blitRenderTargets(nullptr, frameBuffer, false, false);
				Note that in the call above we don't want to copy depth and stencil buffer, but event though default
				FBO contains depth buffer.
				- pass frame buffer to performScreenShot(video::IFrameBuffer*)
				Notes:
				- color buffer is placed under video::EFAP_COLOR_ATTACHMENT0 attachment
				- depth buffer is placed under video::EFAP_DEPTH_ATTACHMENT attachment
			*/

			irr::video::IFrameBuffer* createDefaultFBOForScreenshoting(core::smart_refctd_ptr<IrrlichtDevice> device)
			{
				auto driver = device->getVideoDriver();

				auto createAttachement = [&](bool colorBuffer)
				{
					asset::ICPUImage::SCreationParams imgInfo;
					imgInfo.format = colorBuffer ? asset::EF_R8G8B8A8_SRGB : asset::EF_D24_UNORM_S8_UINT;
					imgInfo.type = asset::ICPUImage::ET_2D;
					imgInfo.extent.width = driver->getScreenSize().Width;
					imgInfo.extent.height = driver->getScreenSize().Height;
					imgInfo.extent.depth = 1u;
					imgInfo.mipLevels = 1u;
					imgInfo.arrayLayers = 1u;
					imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
					imgInfo.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);

					auto image = asset::ICPUImage::create(std::move(imgInfo));
					const auto texelFormatBytesize = getTexelOrBlockBytesize(image->getCreationParameters().format);

					auto texelBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(image->getImageDataSizeInBytes());
					auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::ICPUImage::SBufferCopy>>(1u);
					asset::ICPUImage::SBufferCopy& region = regions->front();

					region.imageSubresource.mipLevel = 0u;
					region.imageSubresource.baseArrayLayer = 0u;
					region.imageSubresource.layerCount = 1u;
					region.bufferOffset = 0u;
					region.bufferRowLength = image->getCreationParameters().extent.width;
					region.bufferImageHeight = 0u;
					region.imageOffset = { 0u, 0u, 0u };
					region.imageExtent = image->getCreationParameters().extent;

					image->setBufferAndRegions(std::move(texelBuffer), regions);

					asset::ICPUImageView::SCreationParams imgViewInfo;
					imgViewInfo.image = std::move(image);
					imgViewInfo.format = colorBuffer ? asset::EF_R8G8B8A8_SRGB : asset::EF_D24_UNORM_S8_UINT;
					imgViewInfo.viewType = asset::IImageView<asset::ICPUImage>::ET_2D;
					imgViewInfo.flags = static_cast<asset::ICPUImageView::E_CREATE_FLAGS>(0u);
					imgViewInfo.subresourceRange.baseArrayLayer = 0u;
					imgViewInfo.subresourceRange.baseMipLevel = 0u;
					imgViewInfo.subresourceRange.layerCount = imgInfo.arrayLayers;
					imgViewInfo.subresourceRange.levelCount = imgInfo.mipLevels;

					auto imageView = asset::ICPUImageView::create(std::move(imgViewInfo));
					auto gpuImageView = driver->getGPUObjectsFromAssets(&imageView.get(), &imageView.get() + 1)->front();

					return std::move(gpuImageView);
				};

				auto gpuImageViewDepthBuffer = createAttachement(false);
				auto gpuImageViewColorBuffer = createAttachement(true);

				auto frameBuffer = driver->addFrameBuffer();
				frameBuffer->attach(video::EFAP_DEPTH_ATTACHMENT, std::move(gpuImageViewDepthBuffer));
				frameBuffer->attach(video::EFAP_COLOR_ATTACHMENT0, std::move(gpuImageViewColorBuffer));

				return frameBuffer;
			};

			/*
				Download mip level image with gpu image usage and save it to IGPUBuffer.
				Because of the fence placed by driver the function stalls the CPU 
				to wait on the GPU to finish, beware of that.
				@see video::IDriverFence
			*/

			//! TODO: HANDLE UNPACK ALIGNMENT
			[[nodiscard]] core::smart_refctd_ptr<video::IDriverFence> downloadImageMipLevel(video::IDriver* driver, video::IGPUImage* source, video::IGPUBuffer* destination, uint32_t sourceMipLevel = 0u, size_t destOffset = 0ull, bool implicitflush = true)
			{
				// will change this, https://github.com/buildaworldnet/IrrlichtBAW/issues/148
				if (isBlockCompressionFormat(source->getCreationParameters().format))
					return nullptr;

				auto extent = source->getMipSize(sourceMipLevel);
				video::IGPUImage::SBufferCopy pRegions[1u] = { {destOffset,extent.x,extent.y,{static_cast<asset::IImage::E_ASPECT_FLAGS>(0u),sourceMipLevel,0u,1u},{0u,0u,0u},{extent.x,extent.y,extent.z}} };
				driver->copyImageToBuffer(source, destination, 1u, pRegions);

				return driver->placeFence(implicitflush);
			}

			/*
				Create a ScreenShot with gpu image usage and save it to a file.
			*/

			bool createScreenShot(core::smart_refctd_ptr<IrrlichtDevice> device, const video::IGPUImageView* gpuImageView, const std::string& outFileName)
			{
				auto driver = device->getVideoDriver();
				auto assetManager = device->getAssetManager();

				auto fetchedImageViewParmas = gpuImageView->getCreationParameters();
				auto gpuImage = fetchedImageViewParmas.image;
				auto fetchedImageParams = gpuImage->getCreationParameters();
				auto image = asset::ICPUImage::create(std::move(fetchedImageParams));

				auto texelBufferRowLength = asset::IImageAssetHandlerBase::calcPitchInBlocks(fetchedImageParams.extent.width * asset::getBlockDimensions(fetchedImageParams.format).X, asset::getTexelOrBlockBytesize(fetchedImageParams.format));

				auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::ICPUImage::SBufferCopy>>(1u);
				asset::ICPUImage::SBufferCopy& region = regions->front();

				region.imageSubresource.mipLevel = 0u;
				region.imageSubresource.baseArrayLayer = 0u;
				region.imageSubresource.layerCount = 1u;
				region.bufferOffset = 0u;
				region.bufferRowLength = texelBufferRowLength;
				region.bufferImageHeight = 0u;
				region.imageOffset = { 0u, 0u, 0u };
				region.imageExtent = image->getCreationParameters().extent;

				video::IDriverMemoryBacked::SDriverMemoryRequirements memoryRequirements;
				memoryRequirements.vulkanReqs.alignment = 64u;
				memoryRequirements.vulkanReqs.memoryTypeBits = 0xffffffffu;
				memoryRequirements.memoryHeapLocation = video::IDriverMemoryAllocation::ESMT_NOT_DEVICE_LOCAL;
				memoryRequirements.mappingCapability = video::IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_READ | video::IDriverMemoryAllocation::EMCF_COHERENT | video::IDriverMemoryAllocation::EMCF_CACHED;
				memoryRequirements.vulkanReqs.size = image->getImageDataSizeInBytes();
				auto destinationBuffer = driver->createGPUBufferOnDedMem(memoryRequirements);

				auto mapPointerGetterFence = downloadImageMipLevel(driver, gpuImage.get(), destinationBuffer.get());

				auto destinationBoundMemory = destinationBuffer->getBoundMemory();
				destinationBoundMemory->mapMemoryRange(video::IDriverMemoryAllocation::EMCAF_READ, { 0u, memoryRequirements.vulkanReqs.size });
				auto texelBuffer = core::make_smart_refctd_ptr<asset::CCustomAllocatorCPUBuffer<core::null_allocator<uint8_t>>>(memoryRequirements.vulkanReqs.size, destinationBoundMemory->getMappedPointer(), core::adopt_memory);

				image->setBufferAndRegions(std::move(texelBuffer), regions);
				auto newCreationParams = image->getCreationParameters();
				
				asset::ICPUImageView::SCreationParams viewParams;
				viewParams.flags = static_cast<asset::ICPUImageView::E_CREATE_FLAGS>(0u);
				viewParams.image = image;
				viewParams.format = newCreationParams.format;
				viewParams.viewType = asset::ICPUImageView::ET_2D;
				viewParams.subresourceRange.baseArrayLayer = 0u;
				viewParams.subresourceRange.layerCount = newCreationParams.arrayLayers;
				viewParams.subresourceRange.baseMipLevel = 0u;
				viewParams.subresourceRange.levelCount = newCreationParams.mipLevels;

				auto imageView = asset::ICPUImageView::create(std::move(viewParams));

				auto tryToWrite = [&](asset::IAsset* asset)
				{
					asset::IAssetWriter::SAssetWriteParams wparams(asset);
					return assetManager->writeAsset(outFileName, wparams);
				};
				
				while (mapPointerGetterFence->waitCPU(1000ull, mapPointerGetterFence->canDeferredFlush()) == video::EDFR_TIMEOUT_EXPIRED) {}

				bool status = tryToWrite(image.get());
				if (!status)
					status = tryToWrite(imageView.get());

				destinationBoundMemory->unmapMemory();
				return status;

			}
		} // namespace ScreenShot
	} // namespace ext
} // namespace irr

#endif // _IRR_EXT_SCREEN_SHOT_INCLUDED_