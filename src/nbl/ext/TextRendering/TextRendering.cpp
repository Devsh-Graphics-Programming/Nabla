
using namespace nbl;
using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::video;

#include "nabla.h"
#include <nbl/ext/TextRendering/TextRendering.h>

namespace nbl
{
namespace ext
{
namespace TextRendering
{

// Generates atlas of MSDF textures for each ASCII character
FontAtlas::FontAtlas(IGPUQueue* queue, ILogicalDevice* device, const std::string& fontFilename, uint32_t msdfWidth, uint32_t msdfHeight, uint32_t charsPerRow)
{
	auto error = FT_Init_FreeType(&library);
	assert(!error);

	error = FT_New_Face(library, "C:\\Windows\\Fonts\\arialbd.ttf", 0, &face);
	assert(!error);

	auto fence = device->createFence(static_cast<nbl::video::IGPUFence::E_CREATE_FLAGS>(0));
	auto commandPool = device->createCommandPool(queue->getFamilyIndex(), nbl::video::IGPUCommandPool::ECF_NONE);
	nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> commandBuffer;
	device->createCommandBuffers(commandPool.get(), nbl::video::IGPUCommandBuffer::EL_PRIMARY, 1u, &commandBuffer);

	commandBuffer->begin(nbl::video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);

	std::vector<nbl::core::smart_refctd_ptr<video::IGPUBuffer>> dataBuffers;
	std::vector<asset::IImage::SBufferCopy> bufferCopies;

	uint32_t characterCount = 0;
	{
		uint32_t atlasGlyphX = 0;
		uint32_t atlasRow = 0;

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

			shape.normalize();
			msdfgen::edgeColoringSimple(shape, 3.0); // TODO figure out what this is
			msdfgen::Bitmap<float, 3> msdfMap(msdfWidth, msdfHeight);
			msdfgen::generateMSDF(msdfMap, shape, 4.0, 1.0, { 4.0, 4.0 });

			uint32_t rowLength = msdfWidth * 4;

			video::IGPUBuffer::SCreationParams bufParams;
			bufParams.size = rowLength * msdfHeight;
			auto data = device->createBuffer(std::move(bufParams));
			auto bufreqs = data->getMemoryReqs();
			bufreqs.memoryTypeBits &= device->getPhysicalDevice()->getUpStreamingMemoryTypeBits();
			auto mem = device->allocate(bufreqs, data.get());

			video::IDeviceMemoryAllocation::MappedMemoryRange mappedMemoryRange(data->getBoundMemory(), 0u, rowLength * msdfHeight);
			device->mapMemory(mappedMemoryRange, video::IDeviceMemoryAllocation::EMCAF_READ);

			auto texelBufferPtr = reinterpret_cast<char*>(data->getBoundMemory()->getMappedPointer());
			for (int y = 0; y < msdfMap.height(); ++y)
			{
				for (int x = 0; x < msdfMap.width(); ++x)
				{
					auto pixel = msdfMap(x, msdfHeight - y);
					texelBufferPtr[(x + y * msdfWidth) * 4 + 0] = msdfgen::pixelFloatToByte(pixel[0]);
					texelBufferPtr[(x + y * msdfWidth) * 4 + 1] = msdfgen::pixelFloatToByte(pixel[1]);
					texelBufferPtr[(x + y * msdfWidth) * 4 + 2] = msdfgen::pixelFloatToByte(pixel[2]);
					texelBufferPtr[(x + y * msdfWidth) * 4 + 3] = 255;
				}
			}

			dataBuffers.push_back(data);

			asset::IImage::SBufferCopy region;
			region.imageSubresource.mipLevel = 0u;
			region.imageSubresource.baseArrayLayer = 0u;
			region.imageSubresource.layerCount = 1u;
			region.bufferOffset = 0u;
			region.bufferRowLength = msdfWidth;
			region.bufferImageHeight = 0u;
			region.imageOffset = { atlasGlyphX * msdfWidth, atlasRow * msdfHeight, 0u };
			region.imageExtent = { msdfWidth, msdfHeight, 1 };

			characterAtlasPosition[int(k)] = { uint16_t(atlasGlyphX * msdfWidth), uint16_t(atlasRow * msdfHeight) };

			dataBuffers.push_back(data);
			bufferCopies.push_back(region);
			characterCount++;
			atlasGlyphX++;
			if (atlasGlyphX == charsPerRow)
			{
				atlasGlyphX = 0;
				atlasRow++;
			}
		}
	}

	commandBuffer->end();

	{
		uint32_t atlasRows = (characterCount + (charsPerRow - 1)) / charsPerRow;

		video::IGPUImage::SCreationParams imgParams;
		imgParams.type = asset::IImage::ET_2D;
		imgParams.format = asset::EF_R8G8B8A8_UNORM;
		imgParams.samples = asset::IImage::ESCF_1_BIT;
		imgParams.extent.width = charsPerRow * msdfWidth;
		imgParams.extent.height = atlasRows * msdfHeight;
		imgParams.extent.depth = 1;
		imgParams.mipLevels = 1;
		imgParams.arrayLayers = 1;

		atlasImage = device->createImage(std::move(imgParams));
	}

	for (uint32_t i = 0; i < characterCount; i++)
	{
		auto data = dataBuffers[i];
		auto region = bufferCopies[i];
		commandBuffer->copyBufferToImage(data.get(), atlasImage.get(), asset::IImage::EL_GENERAL, 1, &region);
	}

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
