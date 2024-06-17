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
	struct Face
	{
		FT_Face face;
	};

	// ! return index to be used later in hatch fill style or text glyph object
	struct MsdfTextureUploadInfo 
	{
		core::smart_refctd_ptr<ICPUBuffer> cpuBuffer;
		uint64_t bufferOffset;
		uint32_t3 imageExtent;
	};


	MsdfTextureUploadInfo generateMsdfForShape(msdfgen::Shape glyph, uint32_t2 msdfExtents, float32_t2 scale, float32_t2 translate);

	TextRenderer(uint32_t in_msdfPixelRange) : msdfPixelRange(in_msdfPixelRange) {
		auto error = FT_Init_FreeType(&m_ftLibrary);
		assert(!error);
	}

	const FT_Library& getFreetypeLibrary() const { return m_ftLibrary; }
	FT_Library& getFreetypeLibrary() { return m_ftLibrary; }

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