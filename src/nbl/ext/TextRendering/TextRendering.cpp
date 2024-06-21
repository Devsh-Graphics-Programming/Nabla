
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
core::smart_refctd_ptr<ICPUBuffer> TextRenderer::generateMSDFForShape(msdfgen::Shape glyph, uint32_t2 msdfExtents, float32_t2 scale, float32_t2 translate)
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

	return std::move(cpuBuf);
}

constexpr double FreeTypeFontScaling = 1.0 / 64.0;

TextRenderer::GlyphMetric TextRenderer::Face::getGlyphMetrics(uint32_t glyphId)
{
	auto slot = getGlyphSlot(glyphId);

	return {
		.advance = float64_t2(slot->advance.x, 0.0) * FreeTypeFontScaling,
		.horizontalBearing = float64_t2(slot->metrics.horiBearingX, slot->metrics.horiBearingY) * FreeTypeFontScaling,
		.size = float64_t2(slot->metrics.width, slot->metrics.height) * FreeTypeFontScaling,
	};
}

float64_t2 ftPoint2(const FT_Vector& vector) {
	return float64_t2(FreeTypeFontScaling * vector.x, FreeTypeFontScaling * vector.y);
}

int ftMoveToMSDF(const FT_Vector* to, void* user) {
	GlyphShapeBuilder* context = reinterpret_cast<GlyphShapeBuilder*>(user);
	context->moveTo(ftPoint2(*to));
	return 0;
}

int ftLineToMSDF(const FT_Vector* to, void* user) {
	GlyphShapeBuilder* context = reinterpret_cast<GlyphShapeBuilder*>(user);
	context->lineTo(ftPoint2(*to));
	return 0;
}

int ftConicToMSDF(const FT_Vector* control, const FT_Vector* to, void* user) {
	GlyphShapeBuilder* context = reinterpret_cast<GlyphShapeBuilder*>(user);
	context->quadratic(ftPoint2(*control), ftPoint2(*to));
	return 0;
}

int ftCubicToMSDF(const FT_Vector* control1, const FT_Vector* control2, const FT_Vector* to, void* user) {
	GlyphShapeBuilder* context = reinterpret_cast<GlyphShapeBuilder*>(user);
	context->cubic(ftPoint2(*control1), ftPoint2(*control2), ftPoint2(*to));
	return 0;
}

msdfgen::Shape TextRenderer::Face::generateGlyphShape(uint32_t glyphId)
{
	auto slot = getGlyphSlot(glyphId);
	
	msdfgen::Shape shape;
	nbl::ext::TextRendering::GlyphShapeBuilder builder(shape);
	FT_Outline_Funcs ftFunctions;
	ftFunctions.move_to = &ftMoveToMSDF;
	ftFunctions.line_to = &ftLineToMSDF;
	ftFunctions.conic_to = &ftConicToMSDF;
	ftFunctions.cubic_to = &ftCubicToMSDF;
	ftFunctions.shift = 0;
	ftFunctions.delta = 0;
	FT_Error error = FT_Outline_Decompose(&face->glyph->outline, &ftFunctions, &builder);
	if (error)
		return msdfgen::Shape();

	builder.finish();
	return shape;
}

core::smart_refctd_ptr<ICPUBuffer> TextRenderer::Face::generateGlyphUploadInfo(TextRenderer* textRenderer, uint32_t glyphId, uint32_t2 msdfExtents)
{
	auto shape = generateGlyphShape(glyphId);

	// Empty shapes should've been filtered sooner
	assert(!shape.contours.empty());

	auto shapeBounds = shape.getBounds();

	auto expansionAmount = textRenderer->msdfPixelRange;
	float32_t2 frameSize = float32_t2(
		(shapeBounds.r - shapeBounds.l),
		(shapeBounds.t - shapeBounds.b)
	);
	float32_t2 scale = float32_t2(
		float(msdfExtents.x - 2.0 * expansionAmount) / (frameSize.x),
		float(msdfExtents.y - 2.0 * expansionAmount) / (frameSize.y)
	);
	float32_t2 translate = float32_t2(-shapeBounds.l + expansionAmount / scale.x, -shapeBounds.b + expansionAmount / scale.y);

	return textRenderer->generateMSDFForShape(shape, msdfExtents, scale, translate);
}

}
}
}
