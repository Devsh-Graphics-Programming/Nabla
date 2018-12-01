// Copyright (C) 2009-2012 Gary Conway
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h


/*
	Author:	Gary Conway (Viper) - co-author of the ZIP file format, Feb 1989,
						 see the story at http://www.idcnet.us/ziphistory.html
	Website:	http://idcnet.us
	Email:		codeslinger@vipergc.com
	Created:	March 1, 2009
	Version:	1.0
	Updated:
*/

#ifndef __C_IMAGE_LOADER_RGB_H_INCLUDED__
#define __C_IMAGE_LOADER_RGB_H_INCLUDED__

// define _IRR_RGB_FILE_INVERTED_IMAGE_ to preserve the inverted format of the RGB file
// commenting this out will invert the inverted image,resulting in the image being upright
#define _IRR_RGB_FILE_INVERTED_IMAGE_

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_RGB_LOADER_

#include "irr/asset/IAssetLoader.h"

namespace irr
{
namespace video
{

//! Surface Loader for Silicon Graphics RGB files
class CImageLoaderRGB : public asset::IAssetLoader
{
    // byte-align structures
#include "irr/irrpack.h"

    // the RGB image file header structure

    struct SRGBHeader
    {
        uint16_t Magic;	// IRIS image file magic number
        uint8_t  Storage;	// Storage format
        uint8_t  BPC;	// Number of bytes per pixel channel
        uint16_t Dimension;	// Number of dimensions
        uint16_t Xsize;	// X size in pixels
        uint16_t Ysize;	// Y size in pixels
        uint16_t Zsize;	// Z size in pixels
        uint32_t Pixmin;	// Minimum pixel value
        uint32_t Pixmax;	// Maximum pixel value
        uint32_t Dummy1;	// ignored
        char Imagename[80];// Image name
        uint32_t Colormap;	// Colormap ID
                            //		char Dummy2[404];// Ignored
    } PACK_STRUCT;

    // Default alignment
#include "irr/irrunpack.h"

    // this structure holds context specific data about the file being loaded.

    typedef struct _RGBdata
    {
        uint8_t *tmp,
            *tmpR,
            *tmpG,
            *tmpB,
            *tmpA;


        uint32_t *StartTable;	// compressed data table, holds file offsets
        uint32_t *LengthTable;	// length for the above data, hold lengths for above
        uint32_t TableLen;		// len of above tables

        SRGBHeader Header;	// define the .rgb file header
        uint32_t ImageSize;
        uint8_t *rgbData;

    public:
        _RGBdata() : tmp(0), tmpR(0), tmpG(0), tmpB(0), tmpA(0),
            StartTable(0), LengthTable(0), TableLen(0), ImageSize(0), rgbData(0)
        {
        }

        ~_RGBdata()
        {
            delete[] tmp;
            delete[] tmpR;
            delete[] tmpG;
            delete[] tmpB;
            delete[] tmpA;
            delete[] StartTable;
            delete[] LengthTable;
            delete[] rgbData;
        }

        bool allocateTemps()
        {
            tmp = tmpR = tmpG = tmpB = tmpA = 0;
            tmp = new uint8_t[Header.Xsize * 256 * Header.BPC];
            if (!tmp)
                return false;

            if (Header.Zsize >= 1)
            {
                tmpR = new uint8_t[Header.Xsize * Header.BPC];
                if (!tmpR)
                    return false;
            }
            if (Header.Zsize >= 2)
            {
                tmpG = new uint8_t[Header.Xsize * Header.BPC];
                if (!tmpG)
                    return false;
            }
            if (Header.Zsize >= 3)
            {
                tmpB = new uint8_t[Header.Xsize * Header.BPC];
                if (!tmpB)
                    return false;
            }
            if (Header.Zsize >= 4)
            {
                tmpA = new uint8_t[Header.Xsize * Header.BPC];
                if (!tmpA)
                    return false;
            }
            return true;
        }
    } rgbStruct;

public:
	//! constructor
	CImageLoaderRGB();

    virtual bool isALoadableFileFormat(io::IReadFile* _file) const override;

    virtual const char** getAssociatedFileExtensions() const override
    {
        static const char* ext[]{ "rgb", "rgba", "sgi", "int", "inta", "bw", nullptr };
        return ext;
    }

    virtual uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_IMAGE; }

    virtual asset::IAsset* loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;

private:

	bool readHeader(io::IReadFile* file, rgbStruct& rgb) const;
	void readRGBrow(uint8_t *buf, int y, int z, io::IReadFile* file, rgbStruct& rgb) const;
	void processFile(io::IReadFile *file, rgbStruct& rgb) const;
	bool checkFormat(io::IReadFile *file, rgbStruct& rgb) const;
	bool readOffsetTables(io::IReadFile* file, rgbStruct& rgb) const;
	void converttoARGB(uint32_t* in, const uint32_t size) const;
};

} // end namespace video
} // end namespace irr

#endif // _IRR_COMPILE_WITH_RGB_LOADER_
#endif // __C_IMAGE_LOADER_RGB_H_INCLUDED__

