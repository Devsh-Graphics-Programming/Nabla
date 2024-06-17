
using namespace nbl;
using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::video;

#include "nabla.h"
#include <nbl/ext/TextRendering/TextRendering.h>

// TODO sticking to using this library?
// #define STB_RECT_PACK_IMPLEMENTATION
//#include <nbl/ext/TextRendering/stb_rect_pack.h>

namespace nbl
{
namespace ext
{
namespace TextRendering
{

// extents is the size of the MSDF that is generated (probably 32x32)
// glyphExtents is the area of the "image" that msdf will consider (used as resizing, for fill patterns should be 8x8)
TextRenderer::MsdfTextureUploadInfo TextRenderer::generateMsdfForShape(msdfgen::Shape glyph, uint32_t2 msdfExtents, float32_t2 scale, float32_t2 translate)
{
	uint32_t glyphW = msdfExtents.x;
	uint32_t glyphH = msdfExtents.y;

	auto shapeBounds = glyph.getBounds();

	msdfgen::edgeColoringSimple(glyph, 3.0); // TODO figure out what this is
	msdfgen::Bitmap<float, 4> msdfMap(glyphW, glyphH);

	msdfgen::generateMTSDF(msdfMap, glyph, msdfPixelRange, { scale.x, scale.y }, { translate.x, translate.y });

	auto cpuBuf = core::make_smart_refctd_ptr<ICPUBuffer>(glyphW * glyphH * sizeof(float) * 4);
	float* data = reinterpret_cast<float*>(cpuBuf->getPointer());
	// TODO: Optimize this: negative values aren't being handled properly by the updateImageViaStagingBuffer function
	for (int y = 0; y < msdfMap.height(); ++y)
	{
		for (int x = 0; x < msdfMap.width(); ++x)
		{
			auto pixel = msdfMap(x, glyphH - 1 - y);
			data[(x + y * glyphW) * 4 + 0] = std::clamp(pixel[0], 0.0f, 1.0f);
			data[(x + y * glyphW) * 4 + 1] = std::clamp(pixel[1], 0.0f, 1.0f);
			data[(x + y * glyphW) * 4 + 2] = std::clamp(pixel[2], 0.0f, 1.0f);
			data[(x + y * glyphW) * 4 + 3] = std::clamp(pixel[3], 0.0f, 1.0f);
		}
	}

	return {
		.cpuBuffer = std::move(cpuBuf),
		.bufferOffset = 0ull,
		.imageExtent = { glyphW, glyphH, 1 },
	};
}



}
}
}
