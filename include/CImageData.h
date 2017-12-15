// Copyright (C) 2017- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_IMAGE_DATA_H_INCLUDED__
#define __C_IMAGE_DATA_H_INCLUDED__

#include "IReferenceCounted.h"
#include "string.h"
#include "SColor.h"

namespace irr
{
namespace video
{

class CImageData : public IReferenceCounted
{
        void*       data;

        uint32_t    minCoord[3];
        uint32_t    maxCoord[3];
        uint32_t    colorFormat     : 24;
        uint32_t    mipLevelHint    : 6;
        uint32_t    unpackAlignment : 2;

    protected:
        virtual ~CImageData()
        {
            if (data)
                free(data);
        }

    public:
        /*
        CImageData()
        {
            data = NULL;

            minCoord[0] = 0;
            minCoord[1] = 0;
            minCoord[2] = 0;

            maxCoord[0] = 0;
            maxCoord[1] = 0;
            maxCoord[2] = 0;

            colorFormat = ECF_UNKNOWN;
            mipLevelHint = 0;
            unpackAlignment = 1;
        }
        */
        CImageData(void* inData, uint32_t inMinCoord[3], uint32_t inMaxCoord[3],
                   const uint32_t& inMipLevel, const ECOLOR_FORMAT& inFmt,
                   const uint32_t& inUnpackLineAlignment=1,
                   const bool& dataAllocatedWithMallocAndCanTake=false)
        {
            memcpy(minCoord,inMinCoord,3*sizeof(uint32_t));
            memcpy(maxCoord,inMaxCoord,3*sizeof(uint32_t));

            mipLevelHint = inMipLevel;
            colorFormat = inFmt;
            unpackAlignment = inUnpackLineAlignment;

            if (inData&&dataAllocatedWithMallocAndCanTake)
                data = inData;
            else
            {
                size_t imgByteSize = getImageDataSizeInBytes();
                data = malloc(imgByteSize);
                if (inData)
                    memcpy(data,inData,imgByteSize);
            }
        }

        //!
        void forgetAboutData() {data = NULL;}

        //! Returns pointer to raw data
        void* getData() {return data;}
        const void* getData() const {return data;}

        //! Returns offset in width,height and depth of image slice.
        const uint32_t* getOffset() const {return minCoord;}
        const uint32_t* getSliceMin() const {return getOffset();}

        //! Returns width,height and depth of image slice.
        const uint32_t* getSliceMax() const {return maxCoord;}

        //!
        uint32_t getSupposedMipLevel() const {return mipLevelHint;}

        //! Returns bits per pixel.
        uint32_t getBitsPerPixel() const
        {
            return getBitsPerPixelFromFormat(static_cast<ECOLOR_FORMAT>(colorFormat));
        }

        //! Returns image data size in bytes
        size_t getImageDataSizeInBytes() const
        {
            size_t size[3] = {maxCoord[0]-minCoord[0],maxCoord[1]-minCoord[1],maxCoord[2]-minCoord[2]};

            size_t lineSize = getPitch();
            return lineSize*size[1]*size[2];
        }

        //! Returns image data size in pixels
        size_t getImageDataSizeInPixels() const
        {
            size_t size[3] = {maxCoord[0]-minCoord[0],maxCoord[1]-minCoord[1],maxCoord[2]-minCoord[2]};
            return size[0]*size[1]*size[2];
        }

        //! Returns the color format
        ECOLOR_FORMAT getColorFormat() const {return static_cast<ECOLOR_FORMAT>(colorFormat);}

        //! Returns pitch of image
        uint32_t getPitch() const
        {
            return (getBitsPerPixel()*(maxCoord[0]-minCoord[0]))/8;
        }

        //!
        uint32_t getPitchIncludingAlignment() const
        {
            return (getPitch()+unpackAlignment-1)/unpackAlignment;
        }

        //!
        uint32_t getUnpackAlignment() const {return unpackAlignment;}
};

} // end namespace video
} // end namespace irr

#endif


