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

struct SGlyphData 
{
	// Offset in pixels from the start of the string
	int offsetX, offsetY;
	uint32_t textureAtlasGlyphIndex;
};

struct StringBoundingBox
{
	int minX, minY, 
		maxX, maxY;
};

class NBL_API TextRenderer
{
public:
	TextRenderer(ILogicalDevice* device, uint32_t maxGlyphCount, uint32_t maxStringCount);

	bool poolString(
		// Offset in pixels from top left in the screen
		int offsetX, int offsetY,
		uint32_t glyphCount,
		SGlyphData const* glyphs,
		uint32_t* outStringOffset,
		uint32_t* outGeometryOffset
	);

	// Creates the pipeline layout and the pipeline for rendering text
	core::smart_refctd_ptr<video::IGPUGraphicsPipeline> createPipeline(
		video::IGPUObjectFromAssetConverter::SParams& cpu2gpuParams,

		// The data in these buffers is provided by the frustum culling system before calling drawText
		core::smart_refctd_ptr<video::IGPUBuffer> visibleStrings,
		core::smart_refctd_ptr<video::IGPUBuffer> visibleStringMvps,
		core::smart_refctd_ptr<video::IGPUBuffer> visibleStringGlyphCount
	);

	// - Expansion compute shader: Prefix sum over glyph counts & Output draw args
	// - Indirect draw for text
	void drawText(core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf);

private:

private:
	ILogicalDevice* m_device;
	
	core::smart_refctd_ptr<video::CPropertyPool<core::allocator, uint32_t, StringBoundingBox>> m_stringDataPropertyPool;
	core::smart_refctd_ptr<video::SubAllocatedDataBufferST<>> m_geomDataBuffer;

	core::smart_refctd_ptr<video::IGPUBuffer> m_visibleStringGlyphCountPrefixSum;
	core::smart_refctd_ptr<video::IGPUBuffer> m_indirectDrawArgs;
	// - 30 bits global glyph ID
	// - 2 bits quad indices (0-3 per glyph)
	core::smart_refctd_ptr<video::IGPUBuffer> m_glyphIndexBuffer;

	core::smart_refctd_ptr<video::IGPUComputePipeline> m_expansionPipeline;
};

}
}
}

#endif