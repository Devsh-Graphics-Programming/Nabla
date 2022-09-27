// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_TEXT_RENDERING_H_INCLUDED_
#define _NBL_EXT_TEXT_RENDERING_H_INCLUDED_

#include "nabla.h"

#include "nbl/video/alloc/SubAllocatedDataBuffer.h"
#include "nbl/video/utilities/CPropertyPool.h"
#include <msdfgen/msdfgen.h>
#include <ft2build.h>
#include <nbl/ext/TextRendering/TextRendering.h>
#include FT_FREETYPE_H
#include FT_OUTLINE_H

using namespace nbl;
using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::video;

namespace nbl
{
namespace ext
{
namespace TextRendering
{

constexpr uint32_t asciiAtlasCharacterCount = (int('~') - int(' ')) + 1;

struct SPixelCoord
{
	uint16_t x, y;
};

class NBL_API FontAtlas
{
public:
	FontAtlas(IGPUQueue* queue, ILogicalDevice* device, const std::string& fontFilename, uint32_t glyphWidth, uint32_t glyphHeight, uint32_t charsPerRow, uint32_t padding);

	FT_Library library;
	FT_Face face;

	std::array<std::vector<SPixelCoord>, asciiAtlasCharacterCount> characterAtlasPosition;
	core::smart_refctd_ptr<video::IGPUImage> atlasImage;
};

struct SGlyphData 
{
	SPixelCoord offsetFromStringMin;
	uint16_t textureAtlasGlyphIndex;
};

struct StringBoundingBox
{
	SPixelCoord min, max;
};

class NBL_API TextRenderer
{
public:
	typedef typename uint32_t size_type;

	TextRenderer(FontAtlas&& atlas, core::smart_refctd_ptr<ILogicalDevice>&& device, uint32_t maxGlyphCount, uint32_t maxStringCount, uint32_t maxGlyphsPerString);

	using pool_size_t = uint32_t;

	using glyph_geometry_pool_t = video::CPropertyPool<core::allocator, uint64_t>;
	// Data stored as SoA in property pool:
	// - Glyph offset
	// - String bounding box 
	// - MVP
	using string_pool_t = video::CPropertyPool<core::allocator, pool_size_t, StringBoundingBox, core::matrix3x4SIMD>;

	struct string_handle_t
	{
		pool_size_t stringAddr = core::address_type_traits<pool_size_t>::invalid_address;
		pool_size_t glyphDataAddr = core::address_type_traits<pool_size_t>::invalid_address;
		pool_size_t glyphCount = 0u;
	};

	template<class Clock = typename std::chrono::steady_clock>
	uint32_t allocateStrings(
		IGPUQueue* queue,
		const std::chrono::time_point<Clock>& maxWaitPoint, // lets have also a version without this (default)
		const uint32_t count, // how many strings
		string_handle_t* handles, // output handles, if `glyphDataAddr` was not primed with invalid_address, allocation will not happen, likewise for `stringDataAddr`
		const char* const* stringData,
		const core::matrix3x4SIMD* transformMatricies,
		const StringBoundingBox* wrappingBoxes = nullptr // optional, to wrap paragraphs
	)
	{
		auto fence = m_device->createFence(static_cast<nbl::video::IGPUFence::E_CREATE_FLAGS>(0));
		auto commandPool = m_device->createCommandPool(queue->getFamilyIndex(), nbl::video::IGPUCommandPool::ECF_NONE);
		nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> commandBuffer;
		m_device->createCommandBuffers(commandPool.get(), nbl::video::IGPUCommandBuffer::EL_PRIMARY, 1u, &commandBuffer);

		commandBuffer->begin(nbl::video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);

		std::vector<std::tuple<pool_size_t, pool_size_t, StringBoundingBox, core::matrix3x4SIMD>> stringDataTuples;
		std::vector<uint32_t> stringIndices;
		std::vector<uint64_t> glyphData;
		std::vector<uint32_t> glyphDataIndices;
		std::vector<core::smart_refctd_ptr<video::IGPUBuffer>> glyphDataBuffers;

		stringDataTuples.resize(count);
		stringIndices.resize(count);
		glyphDataBuffers.resize(count);

		for (uint32_t i = 0; i < count; i++)
		{
			const char* string = stringData[i];
			core::matrix3x4SIMD matrix = transformMatricies ? transformMatricies[i] : core::matrix3x4SIMD();
			StringBoundingBox bbox = wrappingBoxes ? wrappingBoxes[i] : StringBoundingBox{ { 0, 0 }, { std::numeric_limits<uint16_t>::max(), std::numeric_limits<uint16_t>::max() } };

			// TODO: Font size parameter
			auto error = FT_Set_Pixel_Sizes(m_fontAtlas.face, 0, 1);

			uint32_t x = bbox.min.x;
			uint32_t y = bbox.min.y;
			for (const char* stringIt = string; *stringIt != '\0'; stringIt++)
			{
				char k = *stringIt;
				wchar_t unicode = wchar_t(k);
				uint32_t glyphIndex = FT_Get_Char_Index(m_fontAtlas.face, unicode);

				if (glyphIndex == 0 || k < ' ' || k > '~') continue;
								
				auto& characterAtlasMips = m_fontAtlas.characterAtlasPosition[int(k) - int(' ')];
				if (characterAtlasMips.size() == 0) continue;
				// TODO mip selection
				SPixelCoord glyphTableOffset = characterAtlasMips[0];

				error = FT_Load_Glyph(m_fontAtlas.face, glyphIndex, FT_LOAD_NO_BITMAP);
				assert(!error);

				auto& glyph = m_fontAtlas.face->glyph;

				uint32_t offsetX = x + glyph->bitmap_left;
				uint32_t offsetY = y + glyph->bitmap_top;
				uint32_t extentX = glyph->bitmap.width;
				uint32_t extentY = glyph->bitmap.rows;

				// [TODO] Allocate from suballocatedbuffer
				// glyph value:
				// - 12 bit offset X, 12 bit offset Y
				// - 8 bit extent X, 8 bit extent Y
				// - 12 bit atlas UV X, 12 bit atlas UV Y
				uint64_t gp = 0;
				gp |= uint64_t(offsetX) << 52;
				gp |= uint64_t(offsetY) << 40;
				gp |= uint64_t(extentX) << 32;
				gp |= uint64_t(extentY) << 24;
				gp |= uint64_t(glyphTableOffset.x) << 12;
				gp |= uint64_t(glyphTableOffset.y);
				glyphData.push_back(gp);
				glyphDataIndices.push_back(0);

				x += glyph->advance.x >> 6;
			} 

			pool_size_t glyphCount = glyphData.size();

			// [TODO]: Use the upload utilities here
			video::IGPUBuffer::SCreationParams bufParams;
			bufParams.size = glyphCount * sizeof(uint64_t);
			bufParams.usage = asset::IBuffer::EUF_TRANSFER_SRC_BIT;

			auto data = m_device->createBuffer(std::move(bufParams));
			auto bufreqs = data->getMemoryReqs();
			bufreqs.memoryTypeBits &= m_device->getPhysicalDevice()->getUpStreamingMemoryTypeBits();
			auto mem = m_device->allocate(bufreqs, data.get());

			video::IDeviceMemoryAllocation::MappedMemoryRange mappedMemoryRange(data->getBoundMemory(), 0u, bufParams.size);
			m_device->mapMemory(mappedMemoryRange, video::IDeviceMemoryAllocation::EMCAF_READ);
			
			memcpy(
				reinterpret_cast<char*>(data->getBoundMemory()->getMappedPointer()),
				reinterpret_cast<char*>(&glyphData[0]), 
				bufParams.size
			);

			bool res = m_stringDataPropertyPool->allocateProperties(&glyphDataIndices[0], &glyphDataIndices[glyphDataIndices.size()]);
			assert(res);
			pool_size_t glyphAllocationIx = glyphDataIndices[0];

			asset::SBufferCopy region;
			region.srcOffset = 0;
			region.dstOffset = glyphAllocationIx * sizeof(uint64_t) + m_geomDataBuffer->getPropertyMemoryBlock(0).offset;
			region.size = bufParams.size;
			commandBuffer->copyBuffer(data.get(), m_geomDataBuffer->getPropertyMemoryBlock(0).buffer.get(), 1, &region);

			glyphData.clear();
			glyphDataIndices.clear();

			glyphDataBuffers[i] = std::move(data);
			stringDataTuples[i] = std::make_tuple<pool_size_t, pool_size_t, StringBoundingBox, core::matrix3x4SIMD>(
				std::move(glyphAllocationIx), 
				std::move(glyphCount),
				std::move(bbox), 
				std::move(matrix)
			);
			stringIndices[i] = string_pool_t::invalid;
		}

		bool res = m_stringDataPropertyPool->allocateProperties(&stringIndices[0], &stringIndices[stringIndices.size()]);
		assert(res);

		for (uint32_t i = 0; i < count; i++)
		{
			// Write the properties that were allocated
			// [TODO]: Use the upload utilities here
			auto [ glyphAllocationIx, glyphCount, bbox, matrix ] = stringDataTuples[i];
			uint32_t index = stringIndices[i];

			auto writeProperty = [&](uint32_t propIx, uint32_t dataSize, const void* pData)
			{
				auto& buf = m_stringDataPropertyPool->getPropertyMemoryBlock(propIx);
				commandBuffer->updateBuffer(buf.buffer.get(), buf.offset + index * dataSize, dataSize, pData);
			};

			writeProperty(0, sizeof(pool_size_t), &glyphAllocationIx);
			writeProperty(1, sizeof(StringBoundingBox), &bbox);
			writeProperty(2, sizeof(core::matrix3x4SIMD), &matrix);

			// Output string_handle_t
			string_handle_t handle;
			handle.stringAddr = stringIndices[i];
			handle.glyphDataAddr = glyphAllocationIx;
			handle.glyphCount = glyphCount;
			handles[i] = handle;
		}

		commandBuffer->end();

		nbl::video::IGPUQueue::SSubmitInfo submit;
		{
			submit.commandBufferCount = 1u;
			submit.commandBuffers = &commandBuffer.get();
			submit.signalSemaphoreCount = 0u;
			submit.waitSemaphoreCount = 0u;

			queue->submit(1u, &submit, fence.get());
		}

		m_device->blockForFences(1u, &fence.get());
	}

	inline uint32_t allocateStrings(
		IGPUQueue* queue,
		const uint32_t count, // how many strings
		string_handle_t* handles, // output handles, if `glyphDataAddr` was not primed with invalid_address, allocation will not happen, likewise for `stringDataAddr`
		const char* const* stringData,
		const core::matrix3x4SIMD* transformMatricies,
		const StringBoundingBox* wrappingBoxes = nullptr // optional, to wrap paragraphs
	)
	{
		return allocateStrings(queue, GPUEventWrapper::default_wait(), count, handles, stringData, transformMatricies, wrappingBoxes);
	}

	void freeStrings(
		const uint32_t count, // how many strings
		const string_handle_t* handles,
		core::smart_refctd_ptr<IGPUFence>&& fence // for a deferred free
	);

	// One of the sets would be our global DS, and the other would be the user provided visible strings DS
	// layout(set=0, binding=0) Glyph geometry data
	// layout(set=0, binding=1) Index buffer
	// layout(set=1, binding=0) Visible string MVPs
	// layout(set=1, binding=1) Visible string glyph count
	// layout(set=1, binding=2) Cummulative visible string glyph count
	core::smart_refctd_ptr<video::IGPUGraphicsPipeline> createPipeline(
		video::IGPUObjectFromAssetConverter::SParams& cpu2gpuParams,
		nbl::system::ILogger* logger,
		nbl::asset::IAssetManager* assetManager,
		core::smart_refctd_ptr<video::IGPURenderpass> renderpass,
		core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> visibleStringLayout
	);

	// Visible strings are provided by the user's culling system
	void updateVisibleStringDS(
		core::smart_refctd_ptr<video::IGPUDescriptorSet> visibleStringDS,
		core::smart_refctd_ptr<video::IGPUBuffer> visibleStringMvps,
		core::smart_refctd_ptr<video::IGPUBuffer> visibleStringGlyphCounts,
		core::smart_refctd_ptr<video::IGPUBuffer> cumulativeGlyphCount
	);

	// now doing aggregate append
	//void prefixSumHelper(
	//	core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf, 
	//	core::smart_refctd_ptr<video::IGPUDescriptorSet> visibleStringDS, 
	//	core::smart_refctd_ptr<video::IGPUDescriptorSet> indirectDrawDS
	//);
	
	void drawText(
		video::IGPUCommandBuffer* cmdbuf,
		video::IGPUDescriptorSet* visibleStringDS,
		video::IGPUBuffer* indirectDrawArgs,
		video::IGPUPipelineLayout* pipelineLayout
	);

	void drawTextIndexed(
		video::IGPUCommandBuffer* cmdbuf,
		video::IGPUDescriptorSet* visibleStringDS,
		video::IGPUBuffer* indirectDrawArgs,
		video::IGPUPipelineLayout* pipelineLayout
	);

	// TODO should be internal TextRendering.cpp thing
	struct FtFallbackContext {
		msdfgen::Point2 position;
		msdfgen::Shape* shape;
		msdfgen::Contour* contour;
	};

	static double f26dot6ToDouble(float x)
	{
		return (1 / 64. * double(x));
	}

	static msdfgen::Point2 ftPoint2(const FT_Vector& vector) {
		return msdfgen::Point2(f26dot6ToDouble(vector.x), f26dot6ToDouble(vector.y));
	}

	static int ftMoveTo(const FT_Vector* to, void* user) {
		FtFallbackContext* context = reinterpret_cast<FtFallbackContext*>(user);
		if (!(context->contour && context->contour->edges.empty()))
			context->contour = &context->shape->addContour();
		context->position = ftPoint2(*to);
		return 0;
	}

	static int ftLineTo(const FT_Vector* to, void* user) {
		FtFallbackContext* context = reinterpret_cast<FtFallbackContext*>(user);
		msdfgen::Point2 endpoint = ftPoint2(*to);
		if (endpoint != context->position) {
			context->contour->addEdge(new msdfgen::LinearSegment(context->position, endpoint));
			context->position = endpoint;
		}
		return 0;
	}

	static int ftConicTo(const FT_Vector* control, const FT_Vector* to, void* user) {
		FtFallbackContext* context = reinterpret_cast<FtFallbackContext*>(user);
		context->contour->addEdge(new msdfgen::QuadraticSegment(context->position, ftPoint2(*control), ftPoint2(*to)));
		context->position = ftPoint2(*to);
		return 0;
	}

	static int ftCubicTo(const FT_Vector* control1, const FT_Vector* control2, const FT_Vector* to, void* user) {
		FtFallbackContext* context = reinterpret_cast<FtFallbackContext*>(user);
		context->contour->addEdge(new msdfgen::CubicSegment(context->position, ftPoint2(*control1), ftPoint2(*control2), ftPoint2(*to)));
		context->position = ftPoint2(*to);
		return 0;
	}

	static bool getGlyphShape(msdfgen::Shape& shape, FT_Library library, FT_Face face)
	{
		FtFallbackContext context = { };
		context.shape = &shape;
		FT_Outline_Funcs ftFunctions;
		ftFunctions.move_to = &ftMoveTo;
		ftFunctions.line_to = &ftLineTo;
		ftFunctions.conic_to = &ftConicTo;
		ftFunctions.cubic_to = &ftCubicTo;
		ftFunctions.shift = 0;
		ftFunctions.delta = 0;
		FT_Error error = FT_Outline_Decompose(&face->glyph->outline, &ftFunctions, &context);
		if (error)
			return false;
		if (!shape.contours.empty() && shape.contours.back().edges.empty())
			shape.contours.pop_back();
		return true;
	}

private:

private:
	core::smart_refctd_ptr<ILogicalDevice> m_device;

	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> m_globalStringDSLayout;
	core::smart_refctd_ptr<video::IGPUDescriptorSet> m_globalStringDS;

	core::smart_refctd_ptr<glyph_geometry_pool_t> m_geomDataBuffer;
	core::smart_refctd_ptr<string_pool_t> m_stringDataPropertyPool;

	// - 30 bits global glyph ID
	// - 2 bits quad indices (0-3 per glyph)
	core::smart_refctd_ptr<video::IGPUBuffer> m_glyphIndexBuffer;

	core::smart_refctd_ptr<video::IGPUComputePipeline> m_expansionPipeline;
	FontAtlas m_fontAtlas;
};

}
}
}

#endif