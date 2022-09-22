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
private:
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

	TextRenderer(core::smart_refctd_ptr<ILogicalDevice>&& device, uint32_t maxGlyphCount, uint32_t maxStringCount, uint32_t maxGlyphsPerString);

	using pool_size_t = uint32_t;

	using glyph_geometry_pool_t = video::SubAllocatedDataBufferST<pool_size_t>;
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
		const std::chrono::time_point<Clock>& maxWaitPoint, // lets have also a version without this (default)
		const uint32_t count, // how many strings
		string_handle_t* handles, // output handles, if `glyphDataAddr` was not primed with invalid_address, allocation will not happen, likewise for `stringDataAddr`
		const char* const* stringData,
		const core::matrix3x4SIMD* transformMatricies,
		const StringBoundingBox* wrappingBoxes = nullptr // optional, to wrap paragraphs
	);

	inline uint32_t allocateStrings(
		const uint32_t count, // how many strings
		string_handle_t* handles, // output handles, if `glyphDataAddr` was not primed with invalid_address, allocation will not happen, likewise for `stringDataAddr`
		const char* const* stringData,
		const core::matrix3x4SIMD* transformMatricies,
		const StringBoundingBox* wrappingBoxes = nullptr // optional, to wrap paragraphs
	)
	{
		return allocateStrings(GPUEventWrapper::default_wait(), count, handles, stringData, transformMatricies, wrappingBoxes);
	}

	void freeStrings(
		const uint32_t count, // how many strings
		const string_handle_t* handles,
		core::smart_refctd_ptr<IGPUFence>&& fence // for a deferred free
	);

	// One of the sets would be our global DS, and the other would be the user provided visible strings DS
	core::smart_refctd_ptr<video::IGPUGraphicsPipeline> createPipeline(
		video::IGPUObjectFromAssetConverter::SParams& cpu2gpuParams, 
		core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> visibleStringLayout
	);

	core::smart_refctd_ptr<video::IGPUGraphicsPipeline> createPipelineIndexed(
		video::IGPUObjectFromAssetConverter::SParams& cpu2gpuParams, 
		core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> visibleStringLayout
	);

	// Visible strings are provided by the user's culling system
	void updateVisibleStringDS(
		core::smart_refctd_ptr<video::IGPUDescriptorSet> visibleStringDS,
		core::smart_refctd_ptr<video::IGPUBuffer> visibleStrings,
		core::smart_refctd_ptr<video::IGPUBuffer> visibleStringGlyphCounts,
		core::smart_refctd_ptr<video::IGPUBuffer> cumulativeGlyphCount
	);

	void prefixSumHelper(
		core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf, 
		core::smart_refctd_ptr<video::IGPUDescriptorSet> visibleStringDS, 
		core::smart_refctd_ptr<video::IGPUDescriptorSet> indirectDrawDS
	);
	
	void drawText(
		core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf,
		core::smart_refctd_ptr<video::IGPUDescriptorSet> visibleStringDS,
		core::smart_refctd_ptr<video::IGPUBuffer> indirectDrawArgs
	);

	void drawTextIndexed(
		core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf,
		core::smart_refctd_ptr<video::IGPUDescriptorSet> visibleStringDS,
		core::smart_refctd_ptr<video::IGPUBuffer> indirectDrawArgs
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
};

}
}
}

#endif