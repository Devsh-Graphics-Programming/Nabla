
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

// TODO find an alternative for this (we need it because msdfgen consumes the glyphs when doing generateMSDF)
msdfgen::Shape deepCloneShape(const msdfgen::Shape& original) {
	msdfgen::Shape copy;
	for (uint32_t contour = 0; contour < original.contours.size(); contour++)
	{
		msdfgen::Contour c;
		auto originalEdges = original.contours[contour].edges;
		c.edges.reserve(originalEdges.size());

		for (uint32_t edge = 0; edge < originalEdges.size(); edge++)
		{
			auto segment = msdfgen::EdgeHolder();
			segment = originalEdges[edge];
			c.edges.push_back(segment);
		}

		copy.contours.push_back(c);
	}
	return copy;
}

core::smart_refctd_ptr<ICPUBuffer> TextRenderer::generateShapeMSDF(msdfgen::Shape glyph, uint32_t msdfPixelRange, uint32_t2 msdfExtents, uint32_t msdfMipLevels, float32_t2 scale, float32_t2 translate)
{
	uint32_t glyphW = msdfExtents.x;
	uint32_t glyphH = msdfExtents.y;

	uint32_t size = 0;
	auto cpuBuf = core::make_smart_refctd_ptr<ICPUBuffer>(size);
	int8_t* data = reinterpret_cast<int8_t*>(cpuBuf->getPointer());
	
	auto floatToSNORM8 = [](const float fl) -> int8_t
	{
		// we need to invert values because msdfgen assigns positive values for shape interior which is the exact opposite of our convention
		return -1 * (int8_t)(std::clamp(fl * 2.0f - 1.0f, -1.0f, 1.0f) * 127.f);
	};

	msdfgen::edgeColoringSimple(glyph, 3.0);

	for (uint32_t mip = 0; mip < msdfMipLevels; mip++)
	{
		msdfgen::Shape glyphCopy = deepCloneShape(glyph);
		uint32_t mipW = glyphW / (1 << mip);
		uint32_t mipH = glyphH / (1 << mip);

		auto shapeBounds = glyphCopy.getBounds();

		msdfgen::Bitmap<float, 4> msdfMap(mipW, mipH);

		msdfgen::generateMTSDF(msdfMap, glyphCopy, msdfPixelRange, { scale.x, scale.y }, { translate.x, translate.y });

		for (int y = 0; y < mipW; ++y)
		{
			for (int x = 0; x < mipH; ++x)
			{
				auto pixel = msdfMap(x, mipH - 1 - y);
				data[(x + y * mipW) * 4 + 0] = floatToSNORM8(pixel[0]);
				data[(x + y * mipW) * 4 + 1] = floatToSNORM8(pixel[1]);
				data[(x + y * mipW) * 4 + 2] = floatToSNORM8(pixel[2]);
				data[(x + y * mipW) * 4 + 3] = floatToSNORM8(pixel[3]);
			}
		}
		data += mipW * mipH * 4;
	}

	return std::move(cpuBuf);
}

constexpr double FreeTypeFontScaling = 1.0 / 64.0;

FontFace::GlyphMetrics FontFace::getGlyphMetricss(uint32_t glyphId)
{
	auto slot = getGlyphSlot(glyphId);

	return {
		.advance = float64_t2(slot->advance.x, 0.0) * FreeTypeFontScaling,
		.horizontalBearing = float64_t2(slot->metrics.horiBearingX, slot->metrics.horiBearingY) * FreeTypeFontScaling,
		.size = float64_t2(slot->metrics.width, slot->metrics.height) * FreeTypeFontScaling,
	};
}

core::smart_refctd_ptr<ICPUBuffer> FontFace::generateGlyphMSDF(uint32_t msdfPixelRange, uint32_t glyphId, uint32_t2 textureExtents, uint32_t mipLevels)
{
	auto shape = generateGlyphShape(glyphId);

	// Empty shapes should've been filtered sooner
	assert(!shape.contours.empty());

	auto shapeBounds = shape.getBounds();

	float32_t2 frameSize = float32_t2(
		(shapeBounds.r - shapeBounds.l),
		(shapeBounds.t - shapeBounds.b)
	);

	const float32_t2 margin = float32_t2(msdfPixelRange * 2.0f);
	const float32_t2 nonUniformScale = (float32_t2(textureExtents) - margin) / frameSize;
	const float32_t uniformScale = core::min(nonUniformScale.x, nonUniformScale.y);
	
	// Center before: ((shapeBounds.l + shapeBounds.r) * 0.5, (shapeBounds.t + shapeBounds.b) * 0.5)
	// Center after: msdfExtents / 2.0
	// Transformation implementation: Center after = (Center before + Translation) * Scale
	// Plugging in the values and solving for translate yields:
	// Translate = (msdfExtents / (2 * scale)) - ((shapeBounds.l + shapeBounds.r) * 0.5, (shapeBounds.t + shapeBounds.b) * 0.5)
	const float32_t2 shapeSpaceCenter = float32_t2(shapeBounds.l + shapeBounds.r, shapeBounds.t + shapeBounds.b) * float32_t2(0.5);
	const float32_t2 translate = float32_t2(textureExtents) / (float32_t2(2.0) * uniformScale) - shapeSpaceCenter;

	return m_textRenderer->generateShapeMSDF(shape, msdfPixelRange, textureExtents, mipLevels, float32_t2(uniformScale, uniformScale), translate);
}

float32_t2 FontFace::getUV(float32_t2 uv, float32_t2 glyphSize, uint32_t2 textureExtents, uint32_t msdfPixelRange)
{
	// NOTE[Erfan]: I don't know if the calculations here are the best way to do it, but it was the first solution that came to mind to transform glyph uv to actual texture uv
	const float32_t2 margin = float32_t2(msdfPixelRange * 2);
	const float32_t2 nonUniformScale = (float32_t2(textureExtents) - margin) / glyphSize;
	const float32_t uniformScale = core::min(nonUniformScale.x, nonUniformScale.y);

	// after finding the scale we solve this equation to get translate:
	// uniformScale * V + translate = P ---> where V is in [0, GlyphSize] and P is in [0, TextureSize]
	const float32_t2 translate = (float32_t2(textureExtents) / 2.0f) - (uniformScale * glyphSize / 2.0f);

	// transform uv of glyph into position in actual texture.
	const float32_t2 placeInTexture = (uniformScale * (uv * glyphSize) + translate);
	// divide by textureExtents to get uv
	return placeInTexture / float32_t2(textureExtents);
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

msdfgen::Shape FontFace::generateGlyphShape(uint32_t glyphId)
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
	FT_Error error = FT_Outline_Decompose(&m_ftFace->glyph->outline, &ftFunctions, &builder);
	if (error)
		return msdfgen::Shape();

	builder.finish();
	return shape;
}

}
}
}
