
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
	return int(character) - int(FirstGeneratedCharacter);
}

uint32_t intlog2(uint32_t i)
{
	// TODO do this properly
	return 31u - __lzcnt(i);
}

// TODO: Figure out what this is supposed to do
static double f26dot6ToDouble(float x)
{
	return (1 / 64. * double(x));
}

float64_t2 ftPoint2(const FT_Vector& vector) {
	return float64_t2(f26dot6ToDouble(vector.x), f26dot6ToDouble(vector.y));
}

int ftMoveTo(const FT_Vector* to, void* user) {
	GlyphShapeBuilder* context = reinterpret_cast<GlyphShapeBuilder*>(user);
	context->moveTo(ftPoint2(*to));
	return 0;
}
int ftLineTo(const FT_Vector* to, void* user) {
	GlyphShapeBuilder* context = reinterpret_cast<GlyphShapeBuilder*>(user);
	context->lineTo(ftPoint2(*to));
	return 0;
}

int ftConicTo(const FT_Vector* control, const FT_Vector* to, void* user) {
	GlyphShapeBuilder* context = reinterpret_cast<GlyphShapeBuilder*>(user);
	context->quadratic(ftPoint2(*control), ftPoint2(*to));
	return 0;
}

int ftCubicTo(const FT_Vector* control1, const FT_Vector* control2, const FT_Vector* to, void* user) {
	GlyphShapeBuilder* context = reinterpret_cast<GlyphShapeBuilder*>(user);
	context->cubic(ftPoint2(*control1), ftPoint2(*control2), ftPoint2(*to));
	return 0;
}

bool drawFreetypeGlyph(msdfgen::Shape& shape, FT_Library library, FT_Face face)
{
	GlyphShapeBuilder builder(shape);
	FT_Outline_Funcs ftFunctions;
	ftFunctions.move_to = &ftMoveTo;
	ftFunctions.line_to = &ftLineTo;
	ftFunctions.conic_to = &ftConicTo;
	ftFunctions.cubic_to = &ftCubicTo;
	ftFunctions.shift = 0;
	ftFunctions.delta = 0;
	FT_Error error = FT_Outline_Decompose(&face->glyph->outline, &ftFunctions, &builder);
	if (error)
		return false;
	builder.finish();
	return true;
}

asset::IImage::SBufferCopy TextRenderer::copyGlyphShapeToImage(
	IGPUBuffer* scratchBuffer, uint32_t scratchBufferOffset,
	uint32_t glyphWidth, uint32_t glyphHeight,
	msdfgen::Shape shape
)
{
	auto shapeBounds = shape.getBounds();

	uint32_t shapeBoundsWidth = shapeBounds.r - shapeBounds.l;
	uint32_t shapeBoundsHeight = shapeBounds.t - shapeBounds.b;
	msdfgen::edgeColoringSimple(shape, 3.0); // TODO figure out what this is
	msdfgen::Bitmap<float, 3> msdfMap(glyphWidth, glyphHeight);

	float scaleX = (1.0 / float(shapeBoundsWidth)) * glyphWidth;
	float scaleY = (1.0 / float(shapeBoundsHeight)) * glyphHeight;
	msdfgen::generateMSDF(msdfMap, shape, 4.0, { scaleX, scaleY }, msdfgen::Vector2(-shapeBounds.l, -shapeBounds.b));

	auto texelBufferPtr = reinterpret_cast<char*>(scratchBuffer->getBoundMemory().memory->getMappedPointer()) + scratchBufferOffset;
	for (int y = 0; y < msdfMap.height(); ++y)
	{
		for (int x = 0; x < msdfMap.width(); ++x)
		{
			auto pixel = msdfMap(x, glyphHeight - 1 - y);
			texelBufferPtr[(x + y * glyphWidth) * 4 + 0] = msdfgen::pixelFloatToByte(pixel[0]);
			texelBufferPtr[(x + y * glyphWidth) * 4 + 1] = msdfgen::pixelFloatToByte(pixel[1]);
			texelBufferPtr[(x + y * glyphWidth) * 4 + 2] = msdfgen::pixelFloatToByte(pixel[2]);
			texelBufferPtr[(x + y * glyphWidth) * 4 + 3] = 255;
		}
	}

	asset::IImage::SBufferCopy region;
	region.imageSubresource.aspectMask = asset::IImage::EAF_COLOR_BIT;
	region.imageSubresource.mipLevel = 0u;
	region.imageSubresource.baseArrayLayer = 0u;
	region.imageSubresource.layerCount = 1u;
	region.bufferOffset = scratchBufferOffset;
	region.bufferRowLength = glyphWidth;
	region.bufferImageHeight = 0u;
	region.imageExtent = { glyphWidth, glyphHeight, 1 };

	return region;
}

// Generates atlas of MSDF textures for each ASCII character
FontAtlas::FontAtlas(video::IQueue* queue, ILogicalDevice* device, const std::string& fontFilename, uint32_t atlasWidth, uint32_t atlasHeight, uint32_t pixelSizes, uint32_t padding)
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
		// For each represented character
		for (char k = FirstGeneratedCharacter; k <= LastGeneratedCharacter; k++)
		{
			wchar_t unicode = wchar_t(k);
			uint32_t glyphIndex = FT_Get_Char_Index(face, unicode);

			// special case for space as it seems to break msdfgen
			if (glyphIndex == 0 || k == ' ')
				continue;

			error = FT_Load_Glyph(face, glyphIndex, FT_LOAD_NO_SCALE);
			assert(!error);

			msdfgen::Shape shape;
			bool loadedGlyph = drawFreetypeGlyph(shape, library, face);
			assert(loadedGlyph);

			shape.normalize();
			auto shapeBounds = shape.getBounds();

			uint32_t shapeBoundsW = shapeBounds.r - shapeBounds.l;
			uint32_t shapeBoundsH = shapeBounds.t - shapeBounds.b;
			uint32_t glyphW = shapeBoundsW * pixelSizes;
			uint32_t glyphH = shapeBoundsH * pixelSizes;
			uint32_t mips = std::min(intlog2(glyphW), intlog2(glyphH));

			FontAtlasGlyph glyph;
			glyph.width = glyphW;
			glyph.height = glyphH;
			glyph.mips.resize(mips);
			characterAtlasPosition[getCharacterAtlasPositionIx(k)] = glyph;

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

				uint32_t rowLength = mipW * 4;

				video::IGPUBuffer::SCreationParams bufParams;
				bufParams.size = rowLength * mipH;
				bufParams.usage = asset::IBuffer::EUF_TRANSFER_SRC_BIT;
				
				// TODO: Merge with image_upload_utils and use uploadImageViaStagingBuffer 
				auto data = device->createBuffer(std::move(bufParams));
				auto bufreqs = data->getMemoryReqs();
				bufreqs.memoryTypeBits &= device->getPhysicalDevice()->getUpStreamingMemoryTypeBits();
				auto mem = device->allocate(bufreqs, data.get());

				video::IDeviceMemoryAllocation::MemoryRange mappedMemoryRange(data->getBoundMemory().offset, rowLength * mipH);
				void* mappedPtr = data->getBoundMemory().memory->map(mappedMemoryRange, video::IDeviceMemoryAllocation::EMCAF_READ);
				assert(data->getBoundMemory().memory->getMappedPointer() == mappedPtr);

				// Generate MSDF for the current mip and copy it
				dataBufferCopyRegions.push_back(TextRenderer::copyGlyphShapeToImage(data.get(), 0, mipW, mipH, shape));
				dataBuffers.push_back(data);
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

	auto fence = device->createSemaphore(0u);
	auto commandPool = device->createCommandPool(queue->getFamilyIndex(), nbl::video::IGPUCommandPool::CREATE_FLAGS::NONE);
	nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> commandBuffer;
	commandPool->createCommandBuffers(nbl::video::IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { &commandBuffer, 1 });

	commandBuffer->begin(nbl::video::IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

	video::IGPUCommandBuffer::SImageMemoryBarrier<video::IGPUCommandBuffer::SOwnershipTransferBarrier> layoutTransBarrier;
	layoutTransBarrier.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
	layoutTransBarrier.subresourceRange.baseMipLevel = 0u;
	layoutTransBarrier.subresourceRange.levelCount = 1u;
	layoutTransBarrier.subresourceRange.baseArrayLayer = 0u;
	layoutTransBarrier.subresourceRange.layerCount = 1u;

	layoutTransBarrier.barrier.dep.srcAccessMask = static_cast<asset::ACCESS_FLAGS>(0);
	layoutTransBarrier.barrier.dep.dstAccessMask = asset::ACCESS_FLAGS::TRANSFER_WRITE_BIT;
	layoutTransBarrier.oldLayout = asset::IImage::LAYOUT::UNDEFINED;
	layoutTransBarrier.newLayout = asset::IImage::LAYOUT::TRANSFER_DST_OPTIMAL;
	layoutTransBarrier.image = atlasImage.get();

	video::IGPUCommandBuffer::SPipelineBarrierDependencyInfo barrierInfo;
	{
		barrierInfo.imgBarriers = { &layoutTransBarrier, 1 };
	}
	commandBuffer->pipelineBarrier(
		static_cast<asset::E_DEPENDENCY_FLAGS>(0u),
		barrierInfo);

	for (uint32_t i = 0; i < dataBuffers.size(); i ++)
	{
		auto& buf = dataBuffers[i];
		auto& region = dataBufferCopyRegions[i];
		auto& rect = glyphPackedRects[i];

		// rect.id = (int(k) << 8) | i;
		char k = char(rect.id >> 8);
		uint32_t mip = rect.id & 255;
		characterAtlasPosition[getCharacterAtlasPositionIx(k)].mips[mip].position = { uint16_t(rect.x), uint16_t(rect.y) };

		region.imageOffset = { rect.x + padding, rect.y + padding, 0u };
		commandBuffer->copyBufferToImage(buf.get(), atlasImage.get(), asset::IImage::LAYOUT::TRANSFER_DST_OPTIMAL, 1, &region);
	}

	layoutTransBarrier.barrier.dep.srcAccessMask = asset::ACCESS_FLAGS::TRANSFER_READ_BIT;
	layoutTransBarrier.barrier.dep.dstAccessMask = static_cast<asset::ACCESS_FLAGS>(0);
	layoutTransBarrier.oldLayout = asset::IImage::LAYOUT::TRANSFER_DST_OPTIMAL;
	layoutTransBarrier.newLayout = asset::IImage::LAYOUT::GENERAL;
	layoutTransBarrier.image = atlasImage.get();

	commandBuffer->pipelineBarrier(
		static_cast<asset::E_DEPENDENCY_FLAGS>(0u),
		barrierInfo);

	commandBuffer->end();

	nbl::video::IQueue::SSubmitInfo submit;
	{
		nbl::video::IQueue::SSubmitInfo::SCommandBufferInfo commandBufInfo = { commandBuffer.get()};
		submit.commandBuffers = { &commandBufInfo, 1 };

		nbl::video::IQueue::SSubmitInfo::SSemaphoreInfo signalInfo = { fence.get(), 1u };
		submit.signalSemaphores = { &signalInfo, 1 };

		queue->submit({ &submit, 1 });
	}

	video::ISemaphore::SWaitInfo waitInfos = { fence.get(), 1u };
	device->blockForSemaphores({ &waitInfos, 1 });
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
			bindings[0].type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
			bindings[0].count = 1u;
			bindings[0].stageFlags = asset::IShader::ESS_COMPUTE;
			bindings[0].samplers = nullptr;
		}
		{
			bindings[1].binding = 1u;
			bindings[1].type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
			bindings[1].count = 1u;
			bindings[1].stageFlags = asset::IShader::ESS_COMPUTE;
			bindings[1].samplers = nullptr;
		}
		m_globalStringDSLayout =
			m_device->createDescriptorSetLayout(bindings);

		video::IDescriptorPool::SCreateInfo poolCreateInfo = {};
		poolCreateInfo.flags = 
			static_cast<video::IDescriptorPool::E_CREATE_FLAGS>(0);
		poolCreateInfo.maxSets = 1;
		poolCreateInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER)] = 2;

		core::smart_refctd_ptr<video::IDescriptorPool> descriptorPool
			= m_device->createDescriptorPool(std::move(poolCreateInfo));

		m_globalStringDS = descriptorPool->createDescriptorSet(core::smart_refctd_ptr(m_globalStringDSLayout));

		const uint32_t writeDescriptorCount = 2u;

		video::IGPUDescriptorSet::SDescriptorInfo descriptorInfos[writeDescriptorCount];
		video::IGPUDescriptorSet::SWriteDescriptorSet writeDescriptorSets[writeDescriptorCount] = {};

		{
			descriptorInfos[0].info.image.imageLayout = asset::IImage::LAYOUT::GENERAL;
			descriptorInfos[0].info.image.sampler = nullptr;
			// TODO take offset into account?
			descriptorInfos[0].desc = m_geomDataBuffer->getPropertyMemoryBlock(0).buffer;

			writeDescriptorSets[0].dstSet = m_globalStringDS.get();
			writeDescriptorSets[0].binding = 0u;
			writeDescriptorSets[0].arrayElement = 0u;
			writeDescriptorSets[0].count = 1u;
			writeDescriptorSets[0].info = &descriptorInfos[0];
		}

		{
			descriptorInfos[1].info.image.imageLayout = asset::IImage::LAYOUT::GENERAL;
			descriptorInfos[1].info.image.sampler = nullptr;
			descriptorInfos[1].desc = m_glyphIndexBuffer;

			writeDescriptorSets[1].dstSet = m_globalStringDS.get();
			writeDescriptorSets[1].binding = 1u;
			writeDescriptorSets[1].arrayElement = 0u;
			writeDescriptorSets[1].count = 1u;
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
		descriptorInfos[0].info.image.imageLayout = asset::IImage::LAYOUT::GENERAL;
		descriptorInfos[0].info.image.sampler = nullptr;
		descriptorInfos[0].desc = visibleStringMvps;

		writeDescriptorSets[0].dstSet = visibleStringDS.get();
		writeDescriptorSets[0].binding = 0u;
		writeDescriptorSets[0].arrayElement = 0u;
		writeDescriptorSets[0].count = 1u;
		writeDescriptorSets[0].info = &descriptorInfos[0];
	}

	{
		descriptorInfos[1].info.image.imageLayout = asset::IImage::LAYOUT::GENERAL;
		descriptorInfos[1].info.image.sampler = nullptr;
		descriptorInfos[1].desc = visibleStringGlyphOffsets;

		writeDescriptorSets[1].dstSet = visibleStringDS.get();
		writeDescriptorSets[1].binding = 1u;
		writeDescriptorSets[1].arrayElement = 0u;
		writeDescriptorSets[1].count = 1u;
		writeDescriptorSets[1].info = &descriptorInfos[1];
	}

	{
		descriptorInfos[2].info.image.imageLayout = asset::IImage::LAYOUT::GENERAL;
		descriptorInfos[2].info.image.sampler = nullptr;
		descriptorInfos[2].desc = cumulativeGlyphCount;

		writeDescriptorSets[2].dstSet = visibleStringDS.get();
		writeDescriptorSets[2].binding = 2u;
		writeDescriptorSets[2].arrayElement = 0u;
		writeDescriptorSets[2].count = 1u;
		writeDescriptorSets[2].info = &descriptorInfos[2];
	}

	m_device->updateDescriptorSets(writeDescriptorCount, writeDescriptorSets, 0u, nullptr);
}


core::smart_refctd_ptr<video::IGPUGraphicsPipeline> TextRenderer::createPipeline(
	nbl::system::ILogger* logger,
	nbl::asset::IAssetManager* assetManager,
	core::smart_refctd_ptr<video::IGPURenderpass> renderpass,
	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> visibleStringLayout
)
{
	auto loadShader = [&](const char* pathToShader, const video::IGPUShader::E_SHADER_STAGE stage)
	{
		video::IGPUShader::SSpecInfo shaderInfo;

		{
			IAssetLoader::SAssetLoadParams lp = {};
			lp.logger = logger;
			lp.workingDirectory = ""; // virtual root
			auto assetBundle = assetManager->getAsset(pathToShader,lp);
			const auto assets = assetBundle.getContents();
			assert(!assets.empty());

			// lets go straight from ICPUSpecializedShader to IGPUSpecializedShader
			auto cpuShader = IAsset::castDown<ICPUShader>(assets[0]);
			cpuShader->setShaderStage(stage);
			assert(cpuShader);

			auto shader = m_device->createShader(cpuShader.get());
			assert(shader);
			shaderInfo.shader = shader.get();
		}

		return shaderInfo;
	};

	auto pipelineLayout = m_device->createPipelineLayout(
		std::span<const asset::SPushConstantRange>{},
		std::move(m_globalStringDSLayout), 
		std::move(visibleStringLayout)
	);

	asset::SVertexInputParams inputParams = {};
	asset::SPrimitiveAssemblyParams assemblyParams = {};
	asset::SBlendParams blendParams = {};
	asset::SRasterizationParams rasterParams = {};
	rasterParams.depthCompareOp = nbl::asset::ECO_ALWAYS;
	rasterParams.depthWriteEnable = false;
	rasterParams.depthBoundsTestEnable = false;
	rasterParams.faceCullingMode = nbl::asset::EFCM_NONE;

	auto vs = loadShader("text.vert", video::IGPUShader::E_SHADER_STAGE::ESS_VERTEX);
	auto fs = loadShader("text.frag", video::IGPUShader::E_SHADER_STAGE::ESS_FRAGMENT);
	video::IGPUShader::SSpecInfo shaders[2] = { vs, fs };

	video::IGPUGraphicsPipeline::SCreationParams pipelineCreationParams;
	pipelineCreationParams.cached = {
		inputParams,
		assemblyParams,
		rasterParams,
		blendParams,
		0u
	};
	pipelineCreationParams.shaders = shaders;
	pipelineCreationParams.renderpass = renderpass.get();

	core::smart_refctd_ptr<video::IGPUGraphicsPipeline> pipeline;
	m_device->createGraphicsPipelines(nullptr, { &pipelineCreationParams, 1 }, & pipeline);
	assert(pipeline);

	return pipeline;
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
	cmdbuf->drawIndirect({ 0u, core::smart_refctd_ptr<video::IGPUBuffer>(indirectDrawArgs) }, 1, sizeof(uint32_t) * 4);
}

void TextRenderer::drawTextIndexed(
	video::IGPUCommandBuffer* cmdbuf,
	video::IGPUDescriptorSet* visibleStringDS,
	video::IGPUBuffer* indirectDrawArgs,
	video::IGPUPipelineLayout* pipelineLayout
)
{
	cmdbuf->bindIndexBuffer({ 0u, core::smart_refctd_ptr<video::IGPUBuffer>(indirectDrawArgs) }, asset::EIT_32BIT);
	video::IGPUDescriptorSet* ds[2] = { m_globalStringDS.get(), visibleStringDS };
	cmdbuf->bindDescriptorSets(asset::EPBP_GRAPHICS, pipelineLayout, 0, 2, &ds[0]);
	cmdbuf->drawIndexedIndirect({ 0u, core::smart_refctd_ptr<video::IGPUBuffer>(indirectDrawArgs) }, 1, sizeof(uint32_t) * 4);
}

}
}
}
