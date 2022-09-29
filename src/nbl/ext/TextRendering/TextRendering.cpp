
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

TextRenderer::TextRenderer(FontAtlas* fontAtlas, core::smart_refctd_ptr<ILogicalDevice>&& device, uint32_t maxGlyphCount, uint32_t maxStringCount, uint32_t maxGlyphsPerString):
	m_device(std::move(device)), m_fontAtlas(fontAtlas)
{
	m_geomDataBuffer = glyph_geometry_pool_t::create(m_device.get(), maxGlyphsPerString * maxStringCount);
	m_stringDataPropertyPool = string_pool_t::create(m_device.get(), maxStringCount, true);

	{
		video::IGPUBuffer::SCreationParams bufParams;
		bufParams.size = 65536 * sizeof(uint32_t);
		bufParams.usage = core::bitflag<asset::IBuffer::E_USAGE_FLAGS>(asset::IBuffer::EUF_INDEX_BUFFER_BIT) | asset::IBuffer::EUF_STORAGE_BUFFER_BIT;
		m_glyphIndexBuffer = m_device->createBuffer(std::move(bufParams));

		m_device->allocate(m_glyphIndexBuffer->getMemoryReqs(), m_glyphIndexBuffer.get());
	}

	{
		// Global string descriptor set
		const uint32_t bindingCount = 2u;
		video::IGPUDescriptorSetLayout::SBinding bindings[bindingCount];
		{
			bindings[0].binding = 0u;
			bindings[0].type = asset::EDT_STORAGE_BUFFER;
			bindings[0].count = 1u;
			bindings[0].stageFlags = asset::IShader::ESS_COMPUTE;
			bindings[0].samplers = nullptr;
		}
		{
			bindings[1].binding = 1u;
			bindings[1].type = asset::EDT_STORAGE_BUFFER;
			bindings[1].count = 1u;
			bindings[1].stageFlags = asset::IShader::ESS_COMPUTE;
			bindings[1].samplers = nullptr;
		}
		m_globalStringDSLayout =
			m_device->createDescriptorSetLayout(bindings, bindings + bindingCount);

		const uint32_t descriptorPoolSizeCount = 1u;
		video::IDescriptorPool::SDescriptorPoolSize poolSizes[descriptorPoolSizeCount];
		poolSizes[0].type = asset::EDT_STORAGE_BUFFER;
		poolSizes[0].count = 2u;

		video::IDescriptorPool::E_CREATE_FLAGS descriptorPoolFlags =
			static_cast<video::IDescriptorPool::E_CREATE_FLAGS>(0);

		core::smart_refctd_ptr<video::IDescriptorPool> descriptorPool
			= m_device->createDescriptorPool(descriptorPoolFlags, 1,
				descriptorPoolSizeCount, poolSizes);

		m_globalStringDS = m_device->createDescriptorSet(descriptorPool.get(),
			core::smart_refctd_ptr(m_globalStringDSLayout));

		const uint32_t writeDescriptorCount = 2u;

		video::IGPUDescriptorSet::SDescriptorInfo descriptorInfos[writeDescriptorCount];
		video::IGPUDescriptorSet::SWriteDescriptorSet writeDescriptorSets[writeDescriptorCount] = {};

		{
			descriptorInfos[0].image.imageLayout = asset::IImage::EL_GENERAL;
			descriptorInfos[0].image.sampler = nullptr;
			// TODO take offset into account?
			descriptorInfos[0].desc = m_geomDataBuffer->getPropertyMemoryBlock(0).buffer;

			writeDescriptorSets[0].dstSet = m_globalStringDS.get();
			writeDescriptorSets[0].binding = 0u;
			writeDescriptorSets[0].arrayElement = 0u;
			writeDescriptorSets[0].count = 1u;
			writeDescriptorSets[0].descriptorType = asset::EDT_STORAGE_BUFFER;
			writeDescriptorSets[0].info = &descriptorInfos[0];
		}

		{
			descriptorInfos[1].image.imageLayout = asset::IImage::EL_GENERAL;
			descriptorInfos[1].image.sampler = nullptr;
			descriptorInfos[1].desc = m_glyphIndexBuffer;

			writeDescriptorSets[1].dstSet = m_globalStringDS.get();
			writeDescriptorSets[1].binding = 1u;
			writeDescriptorSets[1].arrayElement = 0u;
			writeDescriptorSets[1].count = 1u;
			writeDescriptorSets[1].descriptorType = asset::EDT_STORAGE_BUFFER;
			writeDescriptorSets[1].info = &descriptorInfos[1];
		}

		m_device->updateDescriptorSets(writeDescriptorCount, writeDescriptorSets, 0u, nullptr);
	}

}

void TextRenderer::updateVisibleStringDS(
	core::smart_refctd_ptr<video::IGPUDescriptorSet> visibleStringDS,
	core::smart_refctd_ptr<video::IGPUBuffer> visibleStringMvps,
	core::smart_refctd_ptr<video::IGPUBuffer> visibleStringGlyphOffsets,
	core::smart_refctd_ptr<video::IGPUBuffer> cumulativeGlyphCount
)
{
	const uint32_t writeDescriptorCount = 3u;

	video::IGPUDescriptorSet::SDescriptorInfo descriptorInfos[writeDescriptorCount];
	video::IGPUDescriptorSet::SWriteDescriptorSet writeDescriptorSets[writeDescriptorCount] = {};

	{
		descriptorInfos[0].image.imageLayout = asset::IImage::EL_GENERAL;
		descriptorInfos[0].image.sampler = nullptr;
		descriptorInfos[0].desc = visibleStringMvps;

		writeDescriptorSets[0].dstSet = visibleStringDS.get();
		writeDescriptorSets[0].binding = 0u;
		writeDescriptorSets[0].arrayElement = 0u;
		writeDescriptorSets[0].count = 1u;
		writeDescriptorSets[0].descriptorType = asset::EDT_STORAGE_BUFFER;
		writeDescriptorSets[0].info = &descriptorInfos[0];
	}

	{
		descriptorInfos[1].image.imageLayout = asset::IImage::EL_GENERAL;
		descriptorInfos[1].image.sampler = nullptr;
		descriptorInfos[1].desc = visibleStringGlyphOffsets;

		writeDescriptorSets[1].dstSet = visibleStringDS.get();
		writeDescriptorSets[1].binding = 1u;
		writeDescriptorSets[1].arrayElement = 0u;
		writeDescriptorSets[1].count = 1u;
		writeDescriptorSets[1].descriptorType = asset::EDT_STORAGE_BUFFER;
		writeDescriptorSets[1].info = &descriptorInfos[1];
	}

	{
		descriptorInfos[2].image.imageLayout = asset::IImage::EL_GENERAL;
		descriptorInfos[2].image.sampler = nullptr;
		descriptorInfos[2].desc = cumulativeGlyphCount;

		writeDescriptorSets[2].dstSet = visibleStringDS.get();
		writeDescriptorSets[2].binding = 2u;
		writeDescriptorSets[2].arrayElement = 0u;
		writeDescriptorSets[2].count = 1u;
		writeDescriptorSets[2].descriptorType = asset::EDT_STORAGE_BUFFER;
		writeDescriptorSets[2].info = &descriptorInfos[2];
	}

	m_device->updateDescriptorSets(writeDescriptorCount, writeDescriptorSets, 0u, nullptr);
}


core::smart_refctd_ptr<video::IGPUGraphicsPipeline> TextRenderer::createPipeline(
	video::IGPUObjectFromAssetConverter::SParams& cpu2gpuParams,
	nbl::system::ILogger* logger,
	nbl::asset::IAssetManager* assetManager,
	core::smart_refctd_ptr<video::IGPURenderpass> renderpass,
	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> visibleStringLayout
)
{
	auto loadShader = [&](const char* pathToShader)
	{
		core::smart_refctd_ptr<video::IGPUSpecializedShader> specializedShader = nullptr;
		{
			video::IGPUObjectFromAssetConverter cpu2gpu;
			asset::IAssetLoader::SAssetLoadParams params = {};
			params.logger = logger;
			auto spec = (assetManager->getAsset(pathToShader, params).getContents());
			auto specShader_cpu = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset(pathToShader, params).getContents().begin());
			specializedShader = cpu2gpu.getGPUObjectsFromAssets(&specShader_cpu, &specShader_cpu + 1, cpu2gpuParams)->front();
		}
		assert(specializedShader);

		return specializedShader;
	};

	auto pipelineLayout = m_device->createPipelineLayout(nullptr, nullptr, std::move(m_globalStringDSLayout), std::move(visibleStringLayout));

	asset::SVertexInputParams inputParams;
	inputParams.enabledAttribFlags = 0;
	inputParams.enabledBindingFlags = 0;

	asset::SPrimitiveAssemblyParams assemblyParams;
	assemblyParams.primitiveRestartEnable = false;
	assemblyParams.primitiveType = asset::EPT_TRIANGLE_LIST;
	assemblyParams.tessPatchVertCount = 3u;

	asset::SBlendParams blendParams;
	blendParams.logicOpEnable = false;
	blendParams.logicOp = nbl::asset::ELO_NO_OP;

	asset::SRasterizationParams rasterParams;
	rasterParams.depthCompareOp = nbl::asset::ECO_ALWAYS;
	rasterParams.minSampleShading = 1.f;
	rasterParams.depthWriteEnable = false;
	rasterParams.depthTestEnable = false;
	rasterParams.faceCullingMode = nbl::asset::EFCM_NONE;

	auto vs = loadShader("text.vert");
	auto fs = loadShader("text.frag");
	video::IGPUSpecializedShader* shaders[2] = { vs.get(), fs.get() };

	auto gpuRenderpassIndependentPipeline = m_device->createRenderpassIndependentPipeline
	(
		nullptr,
		std::move(pipelineLayout),
		shaders,
		shaders + 2,
		inputParams,
		blendParams,
		assemblyParams,
		rasterParams
	);

	video::IGPUGraphicsPipeline::SCreationParams pipelineCreationParams;
	pipelineCreationParams.renderpassIndependent = gpuRenderpassIndependentPipeline;
	pipelineCreationParams.renderpass = renderpass;

	return m_device->createGraphicsPipeline(nullptr, std::move(pipelineCreationParams));
}

void TextRenderer::drawText(
	video::IGPUCommandBuffer* cmdbuf,
	video::IGPUDescriptorSet* visibleStringDS,
	video::IGPUBuffer* indirectDrawArgs,
	video::IGPUPipelineLayout* pipelineLayout
)
{
	video::IGPUDescriptorSet* ds[2] = { m_globalStringDS.get(), visibleStringDS };
	cmdbuf->bindDescriptorSets(asset::EPBP_GRAPHICS, pipelineLayout, 0, 2, &ds[0]);
	cmdbuf->drawIndirect(indirectDrawArgs, 0, 1, sizeof(uint32_t) * 4);
}

void TextRenderer::drawTextIndexed(
	video::IGPUCommandBuffer* cmdbuf,
	video::IGPUDescriptorSet* visibleStringDS,
	video::IGPUBuffer* indirectDrawArgs,
	video::IGPUPipelineLayout* pipelineLayout
)
{
	cmdbuf->bindIndexBuffer(m_glyphIndexBuffer.get(), 0, asset::EIT_32BIT);
	video::IGPUDescriptorSet* ds[2] = { m_globalStringDS.get(), visibleStringDS };
	cmdbuf->bindDescriptorSets(asset::EPBP_GRAPHICS, pipelineLayout, 0, 2, &ds[0]);
	cmdbuf->drawIndexedIndirect(indirectDrawArgs, 0, 1, sizeof(uint32_t) * 5);
}

}
}
}
