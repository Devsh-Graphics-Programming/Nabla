// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_TEXT_RENDERING_H_INCLUDED_
#define _NBL_EXT_TEXT_RENDERING_H_INCLUDED_

#include "nabla.h"

#include "nbl/video/alloc/SubAllocatedDataBuffer.h"
#include "nbl/video/utilities/CPropertyPool.h"

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

struct SPixelCoord
{
	uint16_t x, y;
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

	TextRenderer(core::smart_refctd_ptr<ILogicalDevice> device, uint32_t maxGlyphCount, uint32_t maxStringCount, uint32_t maxGlyphsPerString);

	using glyph_geometry_pool_t = video::SubAllocatedDataBufferST<>;
	// Data stored as SoA in property pool:
	// - Glyph offset
	// - String bounding box
	using string_pool_t = video::CPropertyPool<core::allocator, glyph_geometry_pool_t::size_type, StringBoundingBox>;

	struct string_handle_t
	{
		string_pool_t::size_type stringAddr = string_pool_t::invalid_address;
		glyph_geometry_pool_t::size_type glyphDataAddr = glyph_geometry_pool_t::invalid_address;
		glyph_geometry_pool_t::size_type glyphCount = 0u;
	};

	template<class Clock = typename std::chrono::steady_clock>
	uint32_t allocateStrings(
		const std::chrono::time_point<Clock>& maxWaitPoint, // lets have also a version without this (default)
		const uint32_t count, // how many strings
		string_handle_t* handles, // output handles, if `glyphDataAddr` was not primed with invalid_address, allocation will not happen, likewise for `stringDataAddr`
		const char* const* stringData,
		const StringBoundingBox* wrappingBoxes = nullptr // optional, to wrap paragraphs
	);

	inline uint32_t allocateStrings(
		const uint32_t count, // how many strings
		string_handle_t* handles, // output handles, if `glyphDataAddr` was not primed with invalid_address, allocation will not happen, likewise for `stringDataAddr`
		const char* const* stringData,
		const StringBoundingBox* wrappingBoxes = nullptr // optional, to wrap paragraphs
	)
	{
		return allocateStrings(GPUEventWrapper::default_wait(), count, handles, stringData, wrappingBoxes);
	}

	void freeStrings(
		const uint32_t count, // how many strings
		const string_handle_t* handles,
		core::smart_refctd_ptr<IGPUFence>&& fence // for a deferred free
	);

	// Creates the pipeline using the cached pipeline layout
	core::smart_refctd_ptr<video::IGPUGraphicsPipeline> createPipeline(video::IGPUObjectFromAssetConverter::SParams& cpu2gpuParams);

	// The data in these buffers is provided by the frustum culling system before calling drawText
	void updateDescriptorSet(
		core::smart_refctd_ptr<video::IGPUBuffer> visibleStrings,
		core::smart_refctd_ptr<video::IGPUBuffer> visibleStringMvps,
		core::smart_refctd_ptr<video::IGPUBuffer> visibleStringGlyphCount,
		core::smart_refctd_ptr<video::IGPUBuffer> prefixSumOutput,
		core::smart_refctd_ptr<video::IGPUBuffer> indirectDrawArgs
	);

	void drawText(core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf, core::smart_refctd_ptr<video::IGPUBuffer> indirectDrawArgs);

	void drawTextIndexed(core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf, core::smart_refctd_ptr<video::IGPUBuffer> indirectDrawArgs);

private:

private:
	core::smart_refctd_ptr<ILogicalDevice> m_device;

	core::smart_refctd_ptr<video::IGPUPipelineLayout> m_pipelineLayout;
	core::smart_refctd_ptr<video::IGPUDescriptorSet> m_descriptorSet;

	core::smart_refctd_ptr<glyph_geometry_pool_t> m_geomDataBuffer;
	core::smart_refctd_ptr<string_pool_t> m_stringDataPropertyPool;

	// - 30 bits global glyph ID
	// - 2 bits quad indices (0-3 per glyph)
	core::smart_refctd_ptr<video::IGPUBuffer> m_glyphIndexBuffer;

	core::smart_refctd_ptr<video::IGPUComputePipeline> m_expansionPipeline;

	void prefixSumHelper(core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf);
};

}
}
}

#endif