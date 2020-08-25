// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_PRECOMPUTED_DITHER_H_INCLUDED__
#define __IRR_C_PRECOMPUTED_DITHER_H_INCLUDED__

#include "../include/irr/asset/filters/dithering/CDither.h"

namespace irr
{
	namespace asset
	{
		//! A class to apply dithering to an image using precomputed dithering image
		/*
			
		*/

		class CPrecomputedDither : public CDither<CPrecomputedDither>
		{
			public:
				CPrecomputedDither() {}
				virtual ~CPrecomputedDither() {}

				//! State of precomputed dithering class
				/*
					The state requires only input dithering image
					which will be completly used in dithering process
					in full extent of image.
				*/
				
				class CState : protected CDither::CState
				{
					public:
						CState(core::smart_refctd_ptr<asset::ICPUImage> _ditheringImage) 
							: ditheringImage(std::move(_ditheringImage))
						{
							imageData.buffer = ditheringImage->getBuffer();
							const auto extent = ditheringImage->getMipSize();
							imageData.strides = TexelBlockInfo(ditheringImage->getCreationParameters().format).convert3DTexelStridesTo1DByteStrides(extent);
							imageData.extent = extent;
						}

						virtual ~CState() {}

						const ImageData& getImageData() const { return imageData; }

					private:

						core::smart_refctd_ptr<asset::ICPUImage> ditheringImage;
				};

				using state_type = CState;

				//! Get channel texel value from dithered image
				/*
					@param state Input state
					@param pixelCoord Current pixel coordinate of processing input image
					we will be applying dithering to
				*/

				static float get(const state_type* state, const core::vectorSIMDu32& pixelCoord)
				{
					const auto& imageData = state->getImageData();

					// if it has to be float, we should change something there cuz we aren't able to fetch the value due to channels
				}
		};
	}
}

#endif // __IRR_C_PRECOMPUTED_DITHER_H_INCLUDED__