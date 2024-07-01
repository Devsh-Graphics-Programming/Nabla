
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

TextRenderer::Face::GeneratedGlyphShape TextRenderer::Face::generateGlyphUploadInfo(TextRenderer* textRenderer, uint32_t glyphId, uint32_t2 msdfExtents)
{
	auto shape = generateGlyphShape(glyphId);

	// Empty shapes should've been filtered sooner
	assert(!shape.contours.empty());

	auto shapeBounds = shape.getBounds();

	float32_t2 frameSize = float32_t2(
		(shapeBounds.r - shapeBounds.l),
		(shapeBounds.t - shapeBounds.b)
	);
	uint32_t biggerAxis = frameSize.y > frameSize.x ? 1 : 0;
	uint32_t smallerAxis = 1 - biggerAxis;

	auto mulMatrix3 = [](float64_t3x3 a, float64_t3x3 b)
	{
		// TODO: Was getting compilation error when doing transform * vector directly, once
		// we can get that to work use it instead
		glm::highp_dmat3x3 glmA;
		memcpy(&glmA, &a, sizeof(float64_t3x3));
		glm::highp_dmat3x3 glmB;
		memcpy(&glmB, &b, sizeof(float64_t3x3));
		auto glmRes = glmA * glmB;
		float64_t3x3 res;
		memcpy(&res, &glmRes, sizeof(float64_t3x3));
		return res;
	};
	auto mulMatrix3Vec = [](float64_t3x3 transformation, float64_t3 vector)
	{
		float64_t2 transfTranslate = float64_t2(transformation[0][2], transformation[1][2]);
		float64_t2 transfScale = float64_t2(transformation[0][0], transformation[1][1]);

		return (vector.xy + transfTranslate) * transfScale;
	};
	auto translate = [](float64_t2 translation)
	{
		auto transform = float64_t3x3();
		transform[0][0] = 1.0;
		transform[1][1] = 1.0;
		transform[2][2] = 1.0;
		
		transform[0][2] = translation.x;
		transform[1][2] = translation.y;
		return transform;
	};
	auto scale = [](float64_t2 scale)
	{
		auto transform = float64_t3x3();
		transform[0][0] = scale.x;
		transform[1][1] = scale.y;
		transform[2][2] = 1.0;
		return transform;
	};
	auto logStateOfThings = [&](float64_t3x3 transformation, std::string desc)
	{
		auto topLeft = mulMatrix3Vec(transformation, float64_t3(shapeBounds.l, shapeBounds.b, 1.0));
		auto topRight = mulMatrix3Vec(transformation, float64_t3(shapeBounds.r, shapeBounds.b, 1.0));
		auto bottomLeft = mulMatrix3Vec(transformation, float64_t3(shapeBounds.l, shapeBounds.t, 1.0));
		auto bottomRight = mulMatrix3Vec(transformation, float64_t3(shapeBounds.r, shapeBounds.t, 1.0));
		auto center = mulMatrix3Vec(transformation, float64_t3((shapeBounds.l + shapeBounds.r) * 0.5, (shapeBounds.b + shapeBounds.t) * 0.5, 1.0));

		float64_t2 transfTranslate = float64_t2(transformation[0][2], transformation[1][2]);
		float64_t2 transfScale = float64_t2(transformation[0][0], transformation[1][1]);
		float64_t2 totalSize = bottomRight - topLeft;

		printf("- Current step: %s\nTranslate: %f %f Scale: %f %f\nTop left: %f %f Top right: %f %f\nBottom left: %f %f Bottom right: %f %f\nCenter: %f %f Total size: %f %f\n", 
			desc.c_str(),
			transfTranslate.x, transfTranslate.y, 
			transfScale.x, transfScale.y, 
			topLeft.x, topLeft.y, topRight.x, topRight.y,
			bottomLeft.x, bottomLeft.y, bottomRight.x, bottomRight.y,
			center.x, center.y, totalSize.x, totalSize.y);
	};

	float32_t smallerSizeRatio = float(frameSize[smallerAxis]) / float(frameSize[biggerAxis]);
	auto generateTransformationMatrix = [&]()
	{
		// Place object top left at origin (0, 0)
		const float64_t2 originTranslation = float64_t2(-shapeBounds.l, -shapeBounds.b);
		const float64_t3x3 originTransform = translate(originTranslation);

		// Scale object to be 32x32 at most, but preserving aspect ratio
		const float64_t2 objectScaleToMsdf = float64_t2(2.0 / max(frameSize.x, frameSize.y)) *
			(float64_t2(msdfExtents) * 0.5);
		const float64_t3x3 objectScaleToMsdfTransform = scale(objectScaleToMsdf);

		// Translate object to be in the middle of the 32x32 MSDF (might have been moved due to aspect ratio)
		float64_t2 objectCentering = float64_t2(0.0, 0.0);
		objectCentering[smallerAxis] = (1.0 - smallerSizeRatio) * 0.5 * msdfExtents[smallerAxis];
		const float64_t3x3 objectCenteringTransform = translate(objectCentering);

		// Transformations for scaling according to the expansion factor
		const uint32_t expansionAmount = textRenderer->msdfPixelRange;
		const float64_t2 expansionFactor = float32_t2(expansionAmount) / float32_t2(msdfExtents);

		const float64_t2 expansionScaleCentered = -float64_t2(msdfExtents) * 0.5;
		const float64_t2 expansionScale = float64_t2(1.0 - (float64_t2(expansionFactor) * 2.0));

		const float64_t3x3 expansionScaleTransform = scale(expansionScale);
		mulMatrix3(
			translate(-expansionScaleCentered),
			mulMatrix3(scale(expansionScale), translate(expansionScaleCentered))
		);

		logStateOfThings(originTransform, "originTransform - object origin at (0, 0)");
		float64_t3x3 transform = mulMatrix3(objectScaleToMsdfTransform, originTransform);
		logStateOfThings(transform, "objectScaleToMsdfTransform * originTransform - scaled to 32x32, size should be 32 for larger axis");

		transform = mulMatrix3(objectCenteringTransform, transform);
		logStateOfThings(transform, "objectCenteringTransform * objectScaleToMsdfTransform * originTransform - translated to be in the middle, center should be 16, 16, size should be 32 for larger axis");

		transform = mulMatrix3(expansionScaleTransform, transform);
		logStateOfThings(transform, "expansionScaleTransform * objectCenteringTransform * objectScaleToMsdfTransform * originTransform - scaled to the expansionf actor, center should be 16, 16 and size should be 24 for larger axis");
		
		return transform;
	};

	printf("\n");
	auto transformation = generateTransformationMatrix();

	//const float32_t2 shapeSizeAfterExpansion = float64_t2(float(msdfExtents[smallerAxis]) - float(textRenderer->msdfPixelRange) * 2.0);
	//const float32_t2 shapeTranslation =
		//float32_t2(-shapeBounds.l, -shapeBounds.b) +
		//(frameSize / float32_t2(msdfExtents)) * (
			//float32_t2(textRenderer->msdfPixelRange) +
			//float32_t2((1.0 - smallerSizeRatio) * 0.5 * (float(msdfExtents[smallerAxis]) - textRenderer->msdfPixelRange * 2.0))
		//);
	//const float32_t2 shapeScale = float32_t2(max(frameSize.x, frameSize.y)) * float32_t2(shapeSizeAfterExpansion);
	//logStateOfThings(mulMatrix3(scale(shapeScale), translate(shapeTranslation)), "shapeScale * shapeTranslation");

	float32_t2 transfTranslate = float32_t2(transformation[0][2], transformation[1][2]);
	float32_t2 transfScale = float32_t2(transformation[0][0], transformation[1][1]);

	//float32_t2 translate = float32_t2(-shapeBounds.l, -shapeBounds.b);
	//float32_t2 scale = (float32_t2(msdfExtents) / frameSize) * aspectRatioPreservingRatio;

	// float32_t2 aspectRatioPreservingRatio = float32_t2(1.0);
	// aspectRatioPreservingRatio[smallerAxis] = smallerSizeRatio;

	//const float fullWidth = 1.0;
	//const float objectWidth = smallerSizeRatio;
	//const float centeringTranslation = fullWidth * 0.5 - objectWidth * 0.5;
	//const float centeringTranslationObjectSpace = centeringTranslation * frameSize[biggerAxis];
	//translate[smallerAxis] += centeringTranslationObjectSpace;

	return {
		.msdfBitmap = textRenderer->generateMSDFForShape(shape, msdfExtents, transfScale, transfTranslate),
		.smallerSizeRatio = smallerSizeRatio,
		.biggerAxis = biggerAxis,
	};
}

}
}
}
