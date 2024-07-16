// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_TEXT_RENDERING_H_INCLUDED_
#define _NBL_EXT_TEXT_RENDERING_H_INCLUDED_

#include "nabla.h"

#include "nbl/video/utilities/CPropertyPool.h"
#include <msdfgen/msdfgen.h>
#include <ft2build.h>
#include <nbl/ext/TextRendering/TextRendering.h>
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
	struct GlyphMetric
	{
		// These already have the scaling from FreeType's em (1.0 / 64.0) applied

		// Offset that should be applied to the current baseline after this glyph is placed
		float64_t2 advance;
		// Offset that the image of the glyph should be placed from the current baseline start
		float64_t2 horizontalBearing;
		// Size of the glyph in the text line
		float64_t2 size;
	};

	static constexpr asset::E_FORMAT MSDFTextureFormat = asset::E_FORMAT::EF_R8G8B8_SNORM;

	// Spits out CPUBuffer containing the image data in SNORM format
	core::smart_refctd_ptr<ICPUBuffer> generateMSDFForShape(msdfgen::Shape glyph, uint32_t2 msdfExtents, float32_t2 scale, float32_t2 translate);

	struct Face : public nbl::core::IReferenceCounted
	{
	public:
		Face(TextRenderer* textRenderer, std::string path)
		{
			auto error = FT_New_Face(textRenderer->getFreetypeLibrary(), path.c_str(), 0, &face);
			assert(!error);

			hash = std::hash<std::string>{}(path);
		}

		uint32_t getGlyphIndex(wchar_t unicode)
		{
			return FT_Get_Char_Index(face, unicode);
		}


		GlyphMetric getGlyphMetrics(uint32_t glyphId);

		msdfgen::Shape generateGlyphShape(uint32_t glyphId);

		struct GeneratedGlyphShape
		{
			core::smart_refctd_ptr<ICPUBuffer> msdfBitmap;
			float32_t smallerSizeRatio;
			uint32_t biggerAxis;
		};
		GeneratedGlyphShape generateGlyphUploadInfo(TextRenderer* textRenderer, uint32_t glyphId, uint32_t2 msdfExtents);

		// TODO: Should be private
		FT_GlyphSlot getGlyphSlot(uint32_t glyphId)
		{
			auto error = FT_Load_Glyph(face, glyphId, FT_LOAD_NO_SCALE);
			assert(!error);
			return face->glyph;
		}
		FT_Face& getFreetypeFace() { return face; }

		size_t getHash() { return hash; }
	protected:
		FT_Face face;
		size_t hash;
	};

	struct GlyphBox
	{
		float64_t2 topLeft;
		float64_t2 dirU;
		float64_t2 dirV;
		uint32_t glyphIdx;
	};

	TextRenderer(uint32_t in_msdfPixelRange) : msdfPixelRange(in_msdfPixelRange) {
		auto error = FT_Init_FreeType(&m_ftLibrary);
		assert(!error);
	}

	const FT_Library& getFreetypeLibrary() const { return m_ftLibrary; }

	FT_Library& getFreetypeLibrary() { return m_ftLibrary; }

	const uint32_t GetPixelRange() { return msdfPixelRange; }

protected:
	uint32_t msdfPixelRange;
	FT_Library m_ftLibrary;

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
			currentContour->addEdge(new msdfgen::LinearSegment(msdfPoint(lastPosition), msdfPoint(to)));
			lastPosition = to;
		}
	}

	// Continue the last moveTo or lineTo with a quadratic bezier:
	// [last position, control, end]
	void quadratic(const float64_t2 control, const float64_t2 to)
	{
		currentContour->addEdge(new msdfgen::QuadraticSegment(msdfPoint(lastPosition), msdfPoint(control), msdfPoint(to)));
		lastPosition = to;
	}

	// Continue the last moveTo or lineTo with a cubic bezier:
	// [last position, control1, control2, end]
	void cubic(const float64_t2 control1, const float64_t2 control2, const float64_t2 to)
	{
		currentContour->addEdge(new msdfgen::CubicSegment(msdfPoint(lastPosition), msdfPoint(control1), msdfPoint(control2), msdfPoint(to)));
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