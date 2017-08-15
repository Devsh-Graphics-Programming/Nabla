// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "IrrCompileConfig.h"
#include "CTRTextureGouraud.h"

#ifdef _IRR_COMPILE_WITH_SOFTWARE_

namespace irr
{
namespace video
{

class CTRTextureGouraudAdd : public CTRTextureGouraud
{
public:

	//! constructor
	CTRTextureGouraudAdd(IZBuffer* zbuffer);

	//! draws an indexed triangle list
	virtual void drawIndexedTriangleList(S2DVertex* vertices, int32_t vertexCount, const uint16_t* indexList, int32_t triangleCount);

protected:

};

//! constructor
CTRTextureGouraudAdd::CTRTextureGouraudAdd(IZBuffer* zbuffer)
: CTRTextureGouraud(zbuffer)
{
	#ifdef _DEBUG
	setDebugName("CTRTextureGouraudAdd");
	#endif
}


//! draws an indexed triangle list
void CTRTextureGouraudAdd::drawIndexedTriangleList(S2DVertex* vertices, int32_t vertexCount, const uint16_t* indexList, int32_t triangleCount)
{
	const S2DVertex *v1, *v2, *v3;

	uint16_t color;
	float tmpDiv; // temporary division factor
	float longest; // saves the longest span
	int32_t height; // saves height of triangle
	uint16_t* targetSurface; // target pointer where to plot pixels
	int32_t spanEnd; // saves end of spans
	float leftdeltaxf; // amount of pixels to increase on left side of triangle
	float rightdeltaxf; // amount of pixels to increase on right side of triangle
	int32_t leftx, rightx; // position where we are
	float leftxf, rightxf; // same as above, but as float values
	int32_t span; // current span
	uint16_t *hSpanBegin, *hSpanEnd; // pointer used when plotting pixels
	int32_t leftR, leftG, leftB, rightR, rightG, rightB; // color values
	int32_t leftStepR, leftStepG, leftStepB,
		rightStepR, rightStepG, rightStepB; // color steps
	int32_t spanR, spanG, spanB, spanStepR, spanStepG, spanStepB; // color interpolating values while drawing a span.
	int32_t leftTx, rightTx, leftTy, rightTy; // texture interpolating values
	int32_t leftTxStep, rightTxStep, leftTyStep, rightTyStep; // texture interpolating values
	int32_t spanTx, spanTy, spanTxStep, spanTyStep; // values of Texturecoords when drawing a span
	core::rect<int32_t> TriangleRect;

	int32_t leftZValue, rightZValue;
	int32_t leftZStep, rightZStep;
	int32_t spanZValue, spanZStep; // ZValues when drawing a span
	TZBufferType* zTarget, *spanZTarget; // target of ZBuffer;

	lockedSurface = (uint16_t*)RenderTarget->lock();
	lockedZBuffer = ZBuffer->lock();
	lockedTexture = (uint16_t*)Texture->lock();

	for (int32_t i=0; i<triangleCount; ++i)
	{
		v1 = &vertices[*indexList];
		++indexList;
		v2 = &vertices[*indexList];
		++indexList;
		v3 = &vertices[*indexList];
		++indexList;

		// back face culling

		if (BackFaceCullingEnabled)
		{
			int32_t z = ((v3->Pos.X - v1->Pos.X) * (v3->Pos.Y - v2->Pos.Y)) -
				((v3->Pos.Y - v1->Pos.Y) * (v3->Pos.X - v2->Pos.X));

			if (z < 0)
				continue;
		}

		//near plane clipping

		if (v1->ZValue<0 && v2->ZValue<0 && v3->ZValue<0)
			continue;

		// sort for width for inscreen clipping

		if (v1->Pos.X > v2->Pos.X)	swapVertices(&v1, &v2);
		if (v1->Pos.X > v3->Pos.X)	swapVertices(&v1, &v3);
		if (v2->Pos.X > v3->Pos.X)	swapVertices(&v2, &v3);

		if ((v1->Pos.X - v3->Pos.X) == 0)
			continue;

		TriangleRect.UpperLeftCorner.X = v1->Pos.X;
		TriangleRect.LowerRightCorner.X = v3->Pos.X;

		// sort for height for faster drawing.

		if (v1->Pos.Y > v2->Pos.Y)	swapVertices(&v1, &v2);
		if (v1->Pos.Y > v3->Pos.Y)	swapVertices(&v1, &v3);
		if (v2->Pos.Y > v3->Pos.Y)	swapVertices(&v2, &v3);

		TriangleRect.UpperLeftCorner.Y = v1->Pos.Y;
		TriangleRect.LowerRightCorner.Y = v3->Pos.Y;

		if (!TriangleRect.isRectCollided(ViewPortRect))
			continue;

		// calculate height of triangle
		height = v3->Pos.Y - v1->Pos.Y;
		if (!height)
			continue;

		// calculate longest span

		longest = (v2->Pos.Y - v1->Pos.Y) / (float)height * (v3->Pos.X - v1->Pos.X) + (v1->Pos.X - v2->Pos.X);

		spanEnd = v2->Pos.Y;
		span = v1->Pos.Y;
		leftxf = (float)v1->Pos.X;
		rightxf = (float)v1->Pos.X;

		leftZValue = v1->ZValue;
		rightZValue = v1->ZValue;

		leftR = rightR = video::getRed(v1->Color)<<8;
		leftG = rightG = video::getGreen(v1->Color)<<8;
		leftB = rightB = video::getBlue(v1->Color)<<8;
		leftTx = rightTx = v1->TCoords.X;
		leftTy = rightTy = v1->TCoords.Y;

		targetSurface = lockedSurface + span * SurfaceWidth;
		zTarget = lockedZBuffer + span * SurfaceWidth;

		if (longest < 0.0f)
		{
			tmpDiv = 1.0f / (float)(v2->Pos.Y - v1->Pos.Y);
			rightdeltaxf = (v2->Pos.X - v1->Pos.X) * tmpDiv;
			rightZStep = (int32_t)((v2->ZValue - v1->ZValue) * tmpDiv);
			rightStepR = (int32_t)(((int32_t)(video::getRed(v2->Color)<<8) - rightR) * tmpDiv);
			rightStepG = (int32_t)(((int32_t)(video::getGreen(v2->Color)<<8) - rightG) * tmpDiv);
			rightStepB = (int32_t)(((int32_t)(video::getBlue(v2->Color)<<8) - rightB) * tmpDiv);
			rightTxStep = (int32_t)((v2->TCoords.X - rightTx) * tmpDiv);
			rightTyStep = (int32_t)((v2->TCoords.Y - rightTy) * tmpDiv);

			tmpDiv = 1.0f / (float)height;
			leftdeltaxf = (v3->Pos.X - v1->Pos.X) * tmpDiv;
			leftZStep = (int32_t)((v3->ZValue - v1->ZValue) * tmpDiv);
			leftStepR = (int32_t)(((int32_t)(video::getRed(v3->Color)<<8) - leftR) * tmpDiv);
			leftStepG = (int32_t)(((int32_t)(video::getGreen(v3->Color)<<8) - leftG) * tmpDiv);
			leftStepB = (int32_t)(((int32_t)(video::getBlue(v3->Color)<<8) - leftB) * tmpDiv);
			leftTxStep = (int32_t)((v3->TCoords.X - leftTx) * tmpDiv);
			leftTyStep = (int32_t)((v3->TCoords.Y - leftTy) * tmpDiv);
		}
		else
		{
			tmpDiv = 1.0f / (float)height;
			rightdeltaxf = (v3->Pos.X - v1->Pos.X) * tmpDiv;
			rightZStep = (int32_t)((v3->ZValue - v1->ZValue) * tmpDiv);
			rightStepR = (int32_t)(((int32_t)(video::getRed(v3->Color)<<8) - rightR) * tmpDiv);
			rightStepG = (int32_t)(((int32_t)(video::getGreen(v3->Color)<<8) - rightG) * tmpDiv);
			rightStepB = (int32_t)(((int32_t)(video::getBlue(v3->Color)<<8) - rightB) * tmpDiv);
			rightTxStep = (int32_t)((v3->TCoords.X - rightTx) * tmpDiv);
			rightTyStep = (int32_t)((v3->TCoords.Y - rightTy) * tmpDiv);

			tmpDiv = 1.0f / (float)(v2->Pos.Y - v1->Pos.Y);
			leftdeltaxf = (v2->Pos.X - v1->Pos.X) * tmpDiv;
			leftZStep = (int32_t)((v2->ZValue - v1->ZValue) * tmpDiv);
			leftStepR = (int32_t)(((int32_t)(video::getRed(v2->Color)<<8) - leftR) * tmpDiv);
			leftStepG = (int32_t)(((int32_t)(video::getGreen(v2->Color)<<8) - leftG) * tmpDiv);
			leftStepB = (int32_t)(((int32_t)(video::getBlue(v2->Color)<<8) - leftB) * tmpDiv);
			leftTxStep = (int32_t)((v2->TCoords.X - leftTx) * tmpDiv);
			leftTyStep = (int32_t)((v2->TCoords.Y - leftTy) * tmpDiv);
		}


		// do it twice, once for the first half of the triangle,
		// end then for the second half.

		for (int32_t triangleHalf=0; triangleHalf<2; ++triangleHalf)
		{
			if (spanEnd > ViewPortRect.LowerRightCorner.Y)
				spanEnd = ViewPortRect.LowerRightCorner.Y;

			// if the span <0, than we can skip these spans,
			// and proceed to the next spans which are really on the screen.
			if (span < ViewPortRect.UpperLeftCorner.Y)
			{
				// we'll use leftx as temp variable
				if (spanEnd < ViewPortRect.UpperLeftCorner.Y)
				{
					leftx = spanEnd - span;
					span = spanEnd;
				}
				else
				{
					leftx = ViewPortRect.UpperLeftCorner.Y - span;
					span = ViewPortRect.UpperLeftCorner.Y;
				}

				leftxf += leftdeltaxf*leftx;
				rightxf += rightdeltaxf*leftx;
				targetSurface += SurfaceWidth*leftx;
				zTarget += SurfaceWidth*leftx;
				leftZValue += leftZStep*leftx;
				rightZValue += rightZStep*leftx;

				leftR += leftStepR*leftx;
				leftG += leftStepG*leftx;
				leftB += leftStepB*leftx;
				rightR += rightStepR*leftx;
				rightG += rightStepG*leftx;
				rightB += rightStepB*leftx;

				leftTx += leftTxStep*leftx;
				leftTy += leftTyStep*leftx;
				rightTx += rightTxStep*leftx;
				rightTy += rightTyStep*leftx;
			}


			// the main loop. Go through every span and draw it.

			while (span < spanEnd)
			{
				leftx = (int32_t)(leftxf);
				rightx = (int32_t)(rightxf + 0.5f);

				// perform some clipping
				// thanks to a correction by hybrid
				// calculations delayed to correctly propagate to textures etc.
				int32_t tDiffLeft=0, tDiffRight=0;
				if (leftx<ViewPortRect.UpperLeftCorner.X)
					tDiffLeft=ViewPortRect.UpperLeftCorner.X-leftx;
				else
				if (leftx>ViewPortRect.LowerRightCorner.X)
					tDiffLeft=ViewPortRect.LowerRightCorner.X-leftx;

				if (rightx<ViewPortRect.UpperLeftCorner.X)
					tDiffRight=ViewPortRect.UpperLeftCorner.X-rightx;
				else
				if (rightx>ViewPortRect.LowerRightCorner.X)
					tDiffRight=ViewPortRect.LowerRightCorner.X-rightx;

				// draw the span
				if (rightx + tDiffRight - leftx - tDiffLeft)
				{
					tmpDiv = 1.0f / (float)(rightx - leftx);
					spanZStep = (int32_t)((rightZValue - leftZValue) * tmpDiv);
					spanZValue = leftZValue+tDiffLeft*spanZStep;

					spanStepR = (int32_t)((rightR - leftR) * tmpDiv);
					spanR = leftR+tDiffLeft*spanStepR;
					spanStepG = (int32_t)((rightG - leftG) * tmpDiv);
					spanG = leftG+tDiffLeft*spanStepG;
					spanStepB = (int32_t)((rightB - leftB) * tmpDiv);
					spanB = leftB+tDiffLeft*spanStepB;

					spanTxStep = (int32_t)((rightTx - leftTx) * tmpDiv);
					spanTx = leftTx + tDiffLeft*spanTxStep;
					spanTyStep = (int32_t)((rightTy - leftTy) * tmpDiv);
					spanTy = leftTy+tDiffLeft*spanTyStep;

					hSpanBegin = targetSurface + leftx+tDiffLeft;
					spanZTarget = zTarget + leftx+tDiffLeft;
					hSpanEnd = targetSurface + rightx+tDiffRight;

					while (hSpanBegin < hSpanEnd)
					{
						if (spanZValue > *spanZTarget)
						{
							//*spanZTarget = spanZValue;
							color = lockedTexture[((spanTy>>8)&textureYMask) * lockedTextureWidth + ((spanTx>>8)&textureXMask)];

							int32_t basis = *hSpanBegin;
							int32_t r = (video::getRed(basis)<<3) + (video::getRed(color)<<3);
							if (r > 255) r = 255;
							int32_t g = (video::getGreen(basis)<<3) + (video::getGreen(color)<<3);
							if (g > 255) g = 255;
							int32_t b = (video::getBlue(basis)<<3) + (video::getBlue(color)<<3);
							if (b > 255) b = 255;

							*hSpanBegin = video::RGB16(r, g, b);
						}

						spanR += spanStepR;
						spanG += spanStepG;
						spanB += spanStepB;

						spanTx += spanTxStep;
						spanTy += spanTyStep;

						spanZValue += spanZStep;
						++hSpanBegin;
						++spanZTarget;
					}
				}

				leftxf += leftdeltaxf;
				rightxf += rightdeltaxf;
				++span;
				targetSurface += SurfaceWidth;
				zTarget += SurfaceWidth;
				leftZValue += leftZStep;
				rightZValue += rightZStep;

				leftR += leftStepR;
				leftG += leftStepG;
				leftB += leftStepB;
				rightR += rightStepR;
				rightG += rightStepG;
				rightB += rightStepB;

				leftTx += leftTxStep;
				leftTy += leftTyStep;
				rightTx += rightTxStep;
				rightTy += rightTyStep;
			}

			if (triangleHalf>0) // break, we've gout only two halves
				break;


			// setup variables for second half of the triangle.

			if (longest < 0.0f)
			{
				tmpDiv = 1.0f / (v3->Pos.Y - v2->Pos.Y);

				rightdeltaxf = (v3->Pos.X - v2->Pos.X) * tmpDiv;
				rightxf = (float)v2->Pos.X;

				rightZValue = v2->ZValue;
				rightZStep = (int32_t)((v3->ZValue - v2->ZValue) * tmpDiv);

				rightR = video::getRed(v2->Color)<<8;
				rightG = video::getGreen(v2->Color)<<8;
				rightB = video::getBlue(v2->Color)<<8;
				rightStepR = (int32_t)(((int32_t)(video::getRed(v3->Color)<<8) - rightR) * tmpDiv);
				rightStepG = (int32_t)(((int32_t)(video::getGreen(v3->Color)<<8) - rightG) * tmpDiv);
				rightStepB = (int32_t)(((int32_t)(video::getBlue(v3->Color)<<8) - rightB) * tmpDiv);

				rightTx = v2->TCoords.X;
				rightTy = v2->TCoords.Y;
				rightTxStep = (int32_t)((v3->TCoords.X - rightTx) * tmpDiv);
				rightTyStep = (int32_t)((v3->TCoords.Y - rightTy) * tmpDiv);
			}
			else
			{
				tmpDiv = 1.0f / (v3->Pos.Y - v2->Pos.Y);

				leftdeltaxf = (v3->Pos.X - v2->Pos.X) * tmpDiv;
				leftxf = (float)v2->Pos.X;

				leftZValue = v2->ZValue;
				leftZStep = (int32_t)((v3->ZValue - v2->ZValue) * tmpDiv);

				leftR = video::getRed(v2->Color)<<8;
				leftG = video::getGreen(v2->Color)<<8;
				leftB = video::getBlue(v2->Color)<<8;
				leftStepR = (int32_t)(((int32_t)(video::getRed(v3->Color)<<8) - leftR) * tmpDiv);
				leftStepG = (int32_t)(((int32_t)(video::getGreen(v3->Color)<<8) - leftG) * tmpDiv);
				leftStepB = (int32_t)(((int32_t)(video::getBlue(v3->Color)<<8) - leftB) * tmpDiv);

				leftTx = v2->TCoords.X;
				leftTy = v2->TCoords.Y;
				leftTxStep = (int32_t)((v3->TCoords.X - leftTx) * tmpDiv);
				leftTyStep = (int32_t)((v3->TCoords.Y - leftTy) * tmpDiv);
			}


			spanEnd = v3->Pos.Y;
		}

	}

	RenderTarget->unlock();
	ZBuffer->unlock();
	Texture->unlock();
}

} // end namespace video
} // end namespace irr

#endif // _IRR_COMPILE_WITH_SOFTWARE_

namespace irr
{
namespace video
{

ITriangleRenderer* createTriangleRendererTextureGouraudAdd(IZBuffer* zbuffer)
{
	#ifdef _IRR_COMPILE_WITH_SOFTWARE_
	return new CTRTextureGouraudAdd(zbuffer);
	#else
	return 0;
	#endif // _IRR_COMPILE_WITH_SOFTWARE_
}

} // end namespace video
} // end namespace irr



