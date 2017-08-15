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

class CTRGouraudWire : public CTRTextureGouraud
{
public:

	CTRGouraudWire(IZBuffer* zbuffer)
		: CTRTextureGouraud(zbuffer)
	{
		#ifdef _DEBUG
		setDebugName("CTRGouraudWire");
		#endif
	}

	//! draws an indexed triangle list
	virtual void drawIndexedTriangleList(S2DVertex* vertices, int32_t vertexCount, const uint16_t* indexList, int32_t triangleCount)
	{
		const S2DVertex *v1, *v2, *v3;

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
		int32_t leftR, leftG, leftB, rightR, rightG, rightB; // color values
		int32_t leftStepR, leftStepG, leftStepB,
			rightStepR, rightStepG, rightStepB; // color steps

		core::rect<int32_t> TriangleRect;

		int32_t leftZValue, rightZValue;
		int32_t leftZStep, rightZStep;
		TZBufferType* zTarget; // target of ZBuffer;

		lockedSurface = (uint16_t*)RenderTarget->lock();
		lockedZBuffer = ZBuffer->lock();

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

			leftR = rightR = video::getRed(v1->Color)<<11;
			leftG = rightG = video::getGreen(v1->Color)<<11;
			leftB = rightB = video::getBlue(v1->Color)<<11;

			targetSurface = lockedSurface + span * SurfaceWidth;
			zTarget = lockedZBuffer + span * SurfaceWidth;

			if (longest < 0.0f)
			{
				tmpDiv = 1.0f / (float)(v2->Pos.Y - v1->Pos.Y);
				rightdeltaxf = (v2->Pos.X - v1->Pos.X) * tmpDiv;
				rightZStep = (int32_t)((v2->ZValue - v1->ZValue) * tmpDiv);
				rightStepR = (int32_t)(((int32_t)(video::getRed(v2->Color)<<11) - rightR) * tmpDiv);
				rightStepG = (int32_t)(((int32_t)(video::getGreen(v2->Color)<<11) - rightG) * tmpDiv);
				rightStepB = (int32_t)(((int32_t)(video::getBlue(v2->Color)<<11) - rightB) * tmpDiv);

				tmpDiv = 1.0f / (float)height;
				leftdeltaxf = (v3->Pos.X - v1->Pos.X) * tmpDiv;
				leftZStep = (int32_t)((v3->ZValue - v1->ZValue) * tmpDiv);
				leftStepR = (int32_t)(((int32_t)(video::getRed(v3->Color)<<11) - leftR) * tmpDiv);
				leftStepG = (int32_t)(((int32_t)(video::getGreen(v3->Color)<<11) - leftG) * tmpDiv);
				leftStepB = (int32_t)(((int32_t)(video::getBlue(v3->Color)<<11) - leftB) * tmpDiv);
			}
			else
			{
				tmpDiv = 1.0f / (float)height;
				rightdeltaxf = (v3->Pos.X - v1->Pos.X) * tmpDiv;
				rightZStep = (int32_t)((v3->ZValue - v1->ZValue) * tmpDiv);
				rightStepR = (int32_t)(((int32_t)(video::getRed(v3->Color)<<11) - rightR) * tmpDiv);
				rightStepG = (int32_t)(((int32_t)(video::getGreen(v3->Color)<<11) - rightG) * tmpDiv);
				rightStepB = (int32_t)(((int32_t)(video::getBlue(v3->Color)<<11) - rightB) * tmpDiv);

				tmpDiv = 1.0f / (float)(v2->Pos.Y - v1->Pos.Y);
				leftdeltaxf = (v2->Pos.X - v1->Pos.X) * tmpDiv;
				leftZStep = (int32_t)((v2->ZValue - v1->ZValue) * tmpDiv);
				leftStepR = (int32_t)(((int32_t)(video::getRed(v2->Color)<<11) - leftR) * tmpDiv);
				leftStepG = (int32_t)(((int32_t)(video::getGreen(v2->Color)<<11) - leftG) * tmpDiv);
				leftStepB = (int32_t)(((int32_t)(video::getBlue(v2->Color)<<11) - leftB) * tmpDiv);
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
				}


				// the main loop. Go through every span and draw it.

				while (span < spanEnd)
				{
					leftx = (int32_t)(leftxf);
					rightx = (int32_t)(rightxf + 0.5f);

					// perform some clipping

					if (leftx>=ViewPortRect.UpperLeftCorner.X &&
						leftx<=ViewPortRect.LowerRightCorner.X)
					{
						if (leftZValue > *(zTarget + leftx))
						{
							*(zTarget + leftx) = leftZValue;
							*(targetSurface + leftx) = video::RGB16(leftR>>8, leftG>>8, leftB>>8);
						}
					}


					if (rightx>=ViewPortRect.UpperLeftCorner.X &&
						rightx<=ViewPortRect.LowerRightCorner.X)
					{
						if (rightZValue > *(zTarget + rightx))
						{
							*(zTarget + rightx) = rightZValue;
							*(targetSurface + rightx) = video::RGB16(rightR, rightG, rightB);
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

					rightR = video::getRed(v2->Color)<<11;
					rightG = video::getGreen(v2->Color)<<11;
					rightB = video::getBlue(v2->Color)<<11;
					rightStepR = (int32_t)(((int32_t)(video::getRed(v3->Color)<<11) - rightR) * tmpDiv);
					rightStepG = (int32_t)(((int32_t)(video::getGreen(v3->Color)<<11) - rightG) * tmpDiv);
					rightStepB = (int32_t)(((int32_t)(video::getBlue(v3->Color)<<11) - rightB) * tmpDiv);
				}
				else
				{
					tmpDiv = 1.0f / (v3->Pos.Y - v2->Pos.Y);

					leftdeltaxf = (v3->Pos.X - v2->Pos.X) * tmpDiv;
					leftxf = (float)v2->Pos.X;

					leftZValue = v2->ZValue;
					leftZStep = (int32_t)((v3->ZValue - v2->ZValue) * tmpDiv);

					leftR = video::getRed(v2->Color)<<11;
					leftG = video::getGreen(v2->Color)<<11;
					leftB = video::getBlue(v2->Color)<<11;
					leftStepR = (int32_t)(((int32_t)(video::getRed(v3->Color)<<11) - leftR) * tmpDiv);
					leftStepG = (int32_t)(((int32_t)(video::getGreen(v3->Color)<<11) - leftG) * tmpDiv);
					leftStepB = (int32_t)(((int32_t)(video::getBlue(v3->Color)<<11) - leftB) * tmpDiv);
				}


				spanEnd = v3->Pos.Y;
			}

		}

		RenderTarget->unlock();
		ZBuffer->unlock();
	}

};

} // end namespace video
} // end namespace irr

#endif // _IRR_COMPILE_WITH_SOFTWARE_

namespace irr
{
namespace video
{


//! creates a flat triangle renderer
ITriangleRenderer* createTriangleRendererGouraudWire(IZBuffer* zbuffer)
{
	#ifdef _IRR_COMPILE_WITH_SOFTWARE_
	return new CTRGouraudWire(zbuffer);
	#else
	return 0;
	#endif // _IRR_COMPILE_WITH_SOFTWARE_
}

} // end namespace video
} // end namespace irr


