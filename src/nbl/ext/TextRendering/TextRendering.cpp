
using namespace nbl;
using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::video;

#include "nabla.h"
#include <nbl/ext/TextRendering/TextRendering.h>

// TODO sticking to using this library?
#define STB_RECT_PACK_IMPLEMENTATION
#include <nbl/ext/TextRendering/stb_rect_pack.h>

namespace nbl
{
namespace ext
{
namespace TextRendering
{

uint32_t getCharacterAtlasPositionIx(char character)
{
	return int(character) - int(' ');
}

uint32_t intlog2(uint32_t i)
{
	// TODO do this properly
	return 31u - __lzcnt(i);
}

// Generates atlas of MSDF textures for each ASCII character
FontAtlas::FontAtlas(IGPUQueue* queue, ILogicalDevice* device, const std::string& fontFilename, uint32_t atlasWidth, uint32_t atlasHeight, uint32_t pixelSizes, uint32_t padding)
{
	auto error = FT_Init_FreeType(&library);
	assert(!error);

	error = FT_New_Face(library, fontFilename.c_str(), 0, &face);
	assert(!error);

	std::vector<stbrp_rect> glyphRects;
	std::vector<nbl::core::smart_refctd_ptr<video::IGPUBuffer>> dataBuffers;
	std::vector<asset::IImage::SBufferCopy> dataBufferCopyRegions;

	const int maxNodes = 4096 * 2;
	struct stbrp_node nodes[maxNodes];

	stbrp_context glyphPackerCtx;
	stbrp_init_target(&glyphPackerCtx, atlasWidth, atlasHeight, nodes, maxNodes);

	{
		// For each character
		for (char k = ' '; k <= '~'; k++)
		{
			wchar_t unicode = wchar_t(k);
			uint32_t glyphIndex = FT_Get_Char_Index(face, unicode);

			// special case for space as it seems to break msdfgen
			if (glyphIndex == 0 || k == ' ')
				continue;

			error = FT_Load_Glyph(face, glyphIndex, FT_LOAD_NO_SCALE);
			assert(!error);

			msdfgen::Shape shape;
			bool loadedGlyph = nbl::ext::TextRendering::TextRenderer::getGlyphShape(shape, library, face);
			assert(loadedGlyph);

			//shape.normalize();
			auto shapeBounds = shape.getBounds();

			uint32_t glyphW = (shapeBounds.r - shapeBounds.l) * pixelSizes;
			uint32_t glyphH = (shapeBounds.t - shapeBounds.b) * pixelSizes;
			uint32_t mips = std::min(intlog2(glyphW), intlog2(glyphH));

			characterAtlasPosition[getCharacterAtlasPositionIx(k)].resize(mips);

			printf("Generating MSDFs for glyph %c; %i mips; Shape bounds: %f %f %f %f\n", k, mips, shapeBounds.l, shapeBounds.b, shapeBounds.r, shapeBounds.t);
			
			for (uint32_t i = 0; i < mips; i++)
			{
				uint32_t div = 1 << i;
				uint32_t mipW = glyphW / div;
				uint32_t mipH = glyphH / div;

				stbrp_rect rect;
				rect.id = (int(k) << 8) | i;
				rect.w = mipW + padding * 2;
				rect.h = mipH + padding * 2;
				rect.x = 0;
				rect.y = 0;
				rect.was_packed = 0;
				glyphRects.push_back(rect);

				printf("Glyph %c; mip %i: %ix%i\n", k, i, rect.w, rect.h);
				// Generate MSDF for the current mip
				msdfgen::edgeColoringSimple(shape, 3.0); // TODO figure out what this is
				msdfgen::Bitmap<float, 3> msdfMap(mipW, mipH);

				float scaleX = (1.0 / float(shapeBounds.r - shapeBounds.l)) * mipW;
				float scaleY = (1.0 / float(shapeBounds.t - shapeBounds.b)) * mipH;
				msdfgen::generateMSDF(msdfMap, shape, 4.0, { scaleX, scaleY }, 0.0);

				uint32_t rowLength = mipW * 4;

				video::IGPUBuffer::SCreationParams bufParams;
				bufParams.size = rowLength * mipH;
				bufParams.usage = asset::IBuffer::EUF_TRANSFER_SRC_BIT;
				
				// TODO: Merge with image_upload_utils and use uploadImageViaStagingBuffer 
				auto data = device->createBuffer(std::move(bufParams));
				auto bufreqs = data->getMemoryReqs();
				bufreqs.memoryTypeBits &= device->getPhysicalDevice()->getUpStreamingMemoryTypeBits();
				auto mem = device->allocate(bufreqs, data.get());

				video::IDeviceMemoryAllocation::MappedMemoryRange mappedMemoryRange(data->getBoundMemory(), 0u, rowLength * mipH);
				device->mapMemory(mappedMemoryRange, video::IDeviceMemoryAllocation::EMCAF_READ);

				auto texelBufferPtr = reinterpret_cast<char*>(data->getBoundMemory()->getMappedPointer());
				for (int y = 0; y < msdfMap.height(); ++y)
				{
					for (int x = 0; x < msdfMap.width(); ++x)
					{
						auto pixel = msdfMap(x, mipH - 1 - y);
						texelBufferPtr[(x + y * mipW) * 4 + 0] = msdfgen::pixelFloatToByte(pixel[0]);
						texelBufferPtr[(x + y * mipW) * 4 + 1] = msdfgen::pixelFloatToByte(pixel[1]);
						texelBufferPtr[(x + y * mipW) * 4 + 2] = msdfgen::pixelFloatToByte(pixel[2]);
						texelBufferPtr[(x + y * mipW) * 4 + 3] = 255;
					}
				}

				asset::IImage::SBufferCopy region;
				region.imageSubresource.aspectMask = asset::IImage::EAF_COLOR_BIT;
				region.imageSubresource.mipLevel = 0u;
				region.imageSubresource.baseArrayLayer = 0u;
				region.imageSubresource.layerCount = 1u;
				region.bufferOffset = 0u;
				region.bufferRowLength = mipW;
				region.bufferImageHeight = 0u;
				region.imageExtent = { mipW, mipH, 1 };

				dataBuffers.push_back(data);
				dataBufferCopyRegions.push_back(region);
			}

		}
	}

	// Pack the glyphs on the atlas
	int res = stbrp_pack_rects(&glyphPackerCtx, &glyphRects[0], glyphRects.size());
	assert(res);

	auto& glyphPackedRects = glyphRects;
	
	{
		video::IGPUImage::SCreationParams imgParams;
		imgParams.type = asset::IImage::ET_2D;
		imgParams.format = asset::EF_R8G8B8A8_UNORM;
		imgParams.samples = asset::IImage::ESCF_1_BIT;
		imgParams.extent.width = atlasWidth;
		imgParams.extent.height = atlasHeight;
		imgParams.extent.depth = 1;
		imgParams.mipLevels = 1;
		imgParams.arrayLayers = 1;
		imgParams.usage = core::bitflag<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_TRANSFER_DST_BIT) | asset::IImage::EUF_SAMPLED_BIT;

		atlasImage = device->createImage(std::move(imgParams));

		auto imgreqs = atlasImage->getMemoryReqs();
		imgreqs.memoryTypeBits &= device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
		device->allocate(imgreqs, atlasImage.get());
	}

	auto fence = device->createFence(static_cast<nbl::video::IGPUFence::E_CREATE_FLAGS>(0));
	auto commandPool = device->createCommandPool(queue->getFamilyIndex(), nbl::video::IGPUCommandPool::ECF_NONE);
	nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> commandBuffer;
	device->createCommandBuffers(commandPool.get(), nbl::video::IGPUCommandBuffer::EL_PRIMARY, 1u, &commandBuffer);

	commandBuffer->begin(nbl::video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);

	video::IGPUCommandBuffer::SImageMemoryBarrier layoutTransBarrier;
	layoutTransBarrier.srcQueueFamilyIndex = ~0u;
	layoutTransBarrier.dstQueueFamilyIndex = ~0u;
	layoutTransBarrier.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
	layoutTransBarrier.subresourceRange.baseMipLevel = 0u;
	layoutTransBarrier.subresourceRange.levelCount = 1u;
	layoutTransBarrier.subresourceRange.baseArrayLayer = 0u;
	layoutTransBarrier.subresourceRange.layerCount = 1u;

	layoutTransBarrier.barrier.srcAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0);
	layoutTransBarrier.barrier.dstAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
	layoutTransBarrier.oldLayout = asset::IImage::EL_UNDEFINED;
	layoutTransBarrier.newLayout = asset::IImage::EL_TRANSFER_DST_OPTIMAL;
	layoutTransBarrier.image = atlasImage;

	commandBuffer->pipelineBarrier(
		asset::EPSF_TOP_OF_PIPE_BIT,
		asset::EPSF_TRANSFER_BIT,
		static_cast<asset::E_DEPENDENCY_FLAGS>(0u),
		0u, nullptr,
		0u, nullptr,
		1u, &layoutTransBarrier);

	for (uint32_t i = 0; i < dataBuffers.size(); i ++)
	{
		auto& buf = dataBuffers[i];
		auto& region = dataBufferCopyRegions[i];
		auto& rect = glyphPackedRects[i];

		// rect.id = (int(k) << 8) | i;
		char k = char(rect.id >> 8);
		uint32_t mip = rect.id & 255;
		characterAtlasPosition[getCharacterAtlasPositionIx(k)][mip] = { uint16_t(rect.x), uint16_t(rect.y) };

		region.imageOffset = { rect.x + padding, rect.y + padding, 0u };
		commandBuffer->copyBufferToImage(buf.get(), atlasImage.get(), asset::IImage::EL_TRANSFER_DST_OPTIMAL, 1, &region);
	}

	layoutTransBarrier.barrier.srcAccessMask = asset::EAF_TRANSFER_READ_BIT;
	layoutTransBarrier.barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0);
	layoutTransBarrier.oldLayout = asset::IImage::EL_TRANSFER_DST_OPTIMAL;
	layoutTransBarrier.newLayout = asset::IImage::EL_GENERAL;
	layoutTransBarrier.image = atlasImage;

	commandBuffer->pipelineBarrier(
		asset::EPSF_TRANSFER_BIT,
		asset::EPSF_BOTTOM_OF_PIPE_BIT,
		static_cast<asset::E_DEPENDENCY_FLAGS>(0u),
		0u, nullptr,
		0u, nullptr,
		1u, &layoutTransBarrier);

	commandBuffer->end();

	nbl::video::IGPUQueue::SSubmitInfo submit;
	{
		submit.commandBufferCount = 1u;
		submit.commandBuffers = &commandBuffer.get();
		submit.signalSemaphoreCount = 0u;
		submit.waitSemaphoreCount = 0u;

		queue->submit(1u, &submit, fence.get());
	}

	device->blockForFences(1u, &fence.get());
}

TextRenderer::TextRenderer(core::smart_refctd_ptr<ILogicalDevice>&& device, uint32_t maxGlyphCount, uint32_t maxStringCount, uint32_t maxGlyphsPerString):
	m_device(std::move(device))
{
	// [TODO]: Initialize these
	// m_geomDataBuffer = core::make_smart_refctd_ptr<glyph_geometry_pool_t>(device.get());
	// m_stringDataPropertyPool = core::make_smart_refctd_ptr<string_pool_t>(device.get());

	{
		// Global string DS
		const uint32_t bindingCount = 2u;
		video::IGPUDescriptorSetLayout::SBinding bindings[bindingCount];
		{
			// Geometry data buffer SSBO
			bindings[0].binding = 0u;
			bindings[0].type = asset::EDT_STORAGE_IMAGE;
			bindings[0].count = 1u;
			bindings[0].stageFlags = asset::IShader::ESS_ALL;
			bindings[0].samplers = nullptr;

			// String data property pool SSBO
			bindings[1].binding = 1u;
			bindings[1].type = asset::EDT_STORAGE_IMAGE;
			bindings[1].count = 1u;
			bindings[1].stageFlags = asset::IShader::ESS_ALL;
			bindings[1].samplers = nullptr;
		}
		m_globalStringDSLayout =
			device->createDescriptorSetLayout(bindings, bindings + bindingCount);

		const uint32_t descriptorPoolSizeCount = 1u;
		video::IDescriptorPool::SDescriptorPoolSize poolSizes[descriptorPoolSizeCount];
		poolSizes[0].type = asset::EDT_STORAGE_IMAGE;
		poolSizes[0].count = 2u;

		video::IDescriptorPool::E_CREATE_FLAGS descriptorPoolFlags =
			static_cast<video::IDescriptorPool::E_CREATE_FLAGS>(0);

		core::smart_refctd_ptr<video::IDescriptorPool> descriptorPool
			= device->createDescriptorPool(descriptorPoolFlags, 1,
				descriptorPoolSizeCount, poolSizes);

		m_globalStringDS = device->createDescriptorSet(descriptorPool.get(),
			core::smart_refctd_ptr(m_globalStringDSLayout));
	}

}

}
}
}
