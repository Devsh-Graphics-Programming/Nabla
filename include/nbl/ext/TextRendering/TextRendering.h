// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_TEXT_RENDERING_H_INCLUDED_
#define _NBL_EXT_TEXT_RENDERING_H_INCLUDED_

#include "nabla.h"

#include "nbl/video/utilities/CPropertyPool.h"
#include <msdfgen/msdfgen.h>
#include <ft2build.h>
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>
#include FT_FREETYPE_H
#include FT_OUTLINE_H

using namespace nbl;
using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::video;
using namespace nbl::hlsl;

namespace nbl
{
namespace ext
{
namespace TextRendering
{

class TextRenderer : public nbl::core::IReferenceCounted
{
public:

	static constexpr asset::E_FORMAT MSDFTextureFormat = asset::E_FORMAT::EF_R8G8B8A8_SNORM;

	// Spits out CPUBuffer containing the image data in SNORM format
	core::smart_refctd_ptr<ICPUBuffer> generateShapeMSDF(msdfgen::Shape glyph, uint32_t msdfPixelRange, uint32_t2 msdfExtents, float32_t2 scale, float32_t2 translate);

	TextRenderer()
	{
		auto error = FT_Init_FreeType(&m_ftLibrary);
		assert(!error);
	}
	
	// TODO: Remove these here, it's only used for customized tests such as building shapes for hatches
	const FT_Library& getFreetypeLibrary() const { return m_ftLibrary; }
	FT_Library& getFreetypeLibrary() { return m_ftLibrary; }

protected:
	friend class FontFace;
	FT_Library m_ftLibrary;
};

class FontFace : public nbl::core::IReferenceCounted
{
public:
	struct GlyphMetrics
	{
		// Offset that should be applied to the current baseline after this glyph is placed
		float64_t2 advance;
		// Offset that the image of the glyph should be placed from the current baseline start, horizontal refers to horizonral LTR or RTL Languages
		float64_t2 horizontalBearing;
		// Size of the glyph in the text line
		float64_t2 size;
	};

	FontFace(core::smart_refctd_ptr<TextRenderer>&& textRenderer, const std::string& path)
	{
		m_textRenderer = std::move(textRenderer);

		auto error = FT_New_Face(m_textRenderer->m_ftLibrary, path.c_str(), 0, &m_ftFace);
		assert(!error);

		m_hash = std::hash<std::string>{}(path);
	}

	static constexpr uint32_t InvalidGlyphIndex = ~0u;

	uint32_t getGlyphIndex(wchar_t unicode)
	{
		if (m_ftFace == nullptr)
		{
			assert(false);
			return InvalidGlyphIndex;
		}
		return FT_Get_Char_Index(m_ftFace, unicode);
	}

	GlyphMetrics getGlyphMetricss(uint32_t glyphId);

	// returns the cpu buffer for the generated MSDF texture with "TextRenderer::MSDFTextureFormat" format
	// it will place the glyph in the center of msdfExtents considering the margin of msdfPixelRange
	// preserves aspect ratio of the glyph corresponding to metrics of the "glyphId"
	// use the `getUV` to address the glyph in your texture correctly.
	core::smart_refctd_ptr<ICPUBuffer> generateGlyphMSDF(uint32_t msdfPixelRange, uint32_t glyphId, uint32_t2 textureExtents);

	// transforms uv in glyph space to uv in the actual texture
	float32_t2 getUV(float32_t2 uv, float32_t2 glyphSize, uint32_t2 textureExtents, uint32_t msdfPixelRange);

	size_t getHash() { return m_hash; }
	
	// TODO: make these protected, it's only used for customized tests such as building shapes for hatches
	FT_GlyphSlot getGlyphSlot(uint32_t glyphId)
	{
		auto error = FT_Load_Glyph(m_ftFace, glyphId, FT_LOAD_NO_SCALE);
		assert(!error);
		return m_ftFace->glyph;
	}
	FT_Face& getFreetypeFace() { return m_ftFace; }
	msdfgen::Shape generateGlyphShape(uint32_t glyphId);

protected:

	core::smart_refctd_ptr<TextRenderer> m_textRenderer;
	FT_Face m_ftFace;
	size_t m_hash;
};

// Helper class for building an msdfgen shape from a glyph
// The shape can be built like a canvas drawing API (move to, line to, 
// and by adding quadratic & cubic segments)
class GlyphShapeBuilder {
public:
	GlyphShapeBuilder(msdfgen::Shape& createShape) : shape(createShape) {}

	// Start a new line from here
	void moveTo(const float64_t2 to)
	{
		if (!(currentContour && currentContour->edges.empty()))
			currentContour = &shape.addContour();
		lastPosition = to;
	}

	// Continue the last line started with moveTo (could also use the last 
	// position from a lineTo)
	void lineTo(const float64_t2 to)
	{
		if (to != lastPosition) {
			currentContour->addEdge(msdfgen::EdgeHolder(msdfPoint(lastPosition), msdfPoint(to)));
			lastPosition = to;
		}
	}

	// Continue the last moveTo or lineTo with a quadratic bezier:
	// [last position, control, end]
	void quadratic(const float64_t2 control, const float64_t2 to)
	{
		currentContour->addEdge(msdfgen::EdgeHolder(msdfPoint(lastPosition), msdfPoint(control), msdfPoint(to)));
		lastPosition = to;
	}

	// Continue the last moveTo or lineTo with a cubic bezier:
	// [last position, control1, control2, end]
	void cubic(const float64_t2 control1, const float64_t2 control2, const float64_t2 to)
	{
		currentContour->addEdge(msdfgen::EdgeHolder(msdfPoint(lastPosition), msdfPoint(control1), msdfPoint(control2), msdfPoint(to)));
		lastPosition = to;
	}

	void finish()
	{
		if (!shape.contours.empty() && shape.contours.back().edges.empty())
			shape.contours.pop_back();
	}
private:
	msdfgen::Point2 msdfPoint(const float64_t2 point)
	{
		return msdfgen::Point2(point.x, point.y);
	}

	// Shape that is currently being created
	msdfgen::Shape& shape;
	// Set with move to and line to
	float64_t2 lastPosition = float64_t2(0.0);
	// Current contour, used for adding edges
	msdfgen::Contour* currentContour = nullptr;
};

}
}
}

#endif