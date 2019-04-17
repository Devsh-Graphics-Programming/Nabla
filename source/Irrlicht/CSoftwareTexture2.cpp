// Copyright (C) 2002-2012 Nikolaus Gebhardt / Thomas Alten
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "IrrCompileConfig.h"
#ifdef _IRR_COMPILE_WITH_BURNINGSVIDEO_

#include "SoftwareDriver2_compile_config.h"
#include "SoftwareDriver2_helper.h"
#include "CSoftwareTexture2.h"
#include "os.h"

namespace irr
{
namespace video
{

//! constructor
CSoftwareTexture2::CSoftwareTexture2(asset::CImageData* image, const io::path& name, uint32_t flags)
                        : ITexture(IDriverMemoryBacked::SDriverMemoryRequirements{{0,0,0},0,0,0,0},name), MipMapLOD(0), Flags ( flags ),
                            OriginalFormat(asset::EF_UNKNOWN)
{
	#ifdef _IRR_DEBUG
	setDebugName("CSoftwareTexture2");
	#endif

	#ifndef SOFTWARE_DRIVER_2_MIPMAPPING
		Flags &= ~GEN_MIPMAP;
	#endif

	memset32 ( MipMap, 0, sizeof ( MipMap ) );

	if (image)
	{
		OrigSize = *reinterpret_cast<const core::dimension2du*>(image->getSliceMax());
		OriginalFormat = image->getColorFormat();

		CImage* tmpImg = new CImage(OriginalFormat,OrigSize,image->getData(),false);

		core::setbit_cond(Flags,
				image->getColorFormat () == asset::EF_B8G8R8A8_UNORM ||
				image->getColorFormat () == asset::EF_A1R5G5B5_UNORM_PACK16,
				HAS_ALPHA);

		core::dimension2d<uint32_t> optSize(
				OrigSize.getOptimalSize( 0 != ( Flags & NP2_SIZE ),
				false, false,
				( Flags & NP2_SIZE ) ? SOFTWARE_DRIVER_2_TEXTURE_MAXSIZE : 0)
			);


        MipMap[0] = new CImage(BURNINGSHADER_COLOR_FORMAT, optSize);
		if ( OrigSize == optSize )
		{
			tmpImg->copyTo(MipMap[0]);
		}
		else
		{
			char buf[256];
			core::stringw showName ( name );
			snprintf ( buf, 256, "Burningvideo: Warning Texture %ls reformat %dx%d -> %dx%d,%d",
							showName.c_str(),
							OrigSize.Width, OrigSize.Height, optSize.Width, optSize.Height,
							BURNINGSHADER_COLOR_FORMAT
						);

			OrigSize = optSize;
			os::Printer::log ( buf, ELL_WARNING );
			tmpImg->copyToScalingBoxFilter ( MipMap[0],0, false );
		}
		tmpImg->drop();

		OrigImageDataSizeInPixels = (float) 0.3f * MipMap[0]->getImageDataSizeInPixels();
	}

	regenerateMipMapLevels();
}


//! destructor
CSoftwareTexture2::~CSoftwareTexture2()
{
	for ( int32_t i = 0; i!= SOFTWARE_DRIVER_2_MIPMAPPING_MAX; ++i )
	{
		if ( MipMap[i] )
			MipMap[i]->drop();
	}
}


//! Regenerates the mip map levels of the texture. Useful after locking and
//! modifying the texture
void CSoftwareTexture2::regenerateMipMapLevels()
{
    os::Printer::log("DevSH says NO!", ELL_ERROR);
		return;
}


} // end namespace video
} // end namespace irr

#endif // _IRR_COMPILE_WITH_BURNINGSVIDEO_
