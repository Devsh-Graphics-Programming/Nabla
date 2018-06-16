// Copyright (C) 2016 Mateusz "DeVsh" Kielan
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __C_MULTI_BUFFERED_INTERFACE_BLOCK_H_INCLUDED__
#define __C_MULTI_BUFFERED_INTERFACE_BLOCK_H_INCLUDED__

#include "irrBaseClasses.h"
#include "IGPUMappedBuffer.h"
#include "IVideoDriver.h"
#include <vector>

namespace irr
{


namespace video
{


template<class BLOCK_STRUCT, size_t BUFFER_COUNT=1>
class CMultiBufferedInterfaceBlock : public virtual IReferenceCounted, public TotalInterface //maybe make non ref-counted in the future
{
        CMultiBufferedInterfaceBlock(IGPUBuffer* inBuffer, const size_t inOffsets[BUFFER_COUNT]) : underlyingBuffer(inBuffer)
        {
            underlyingBuffer->grab();
            memcpy(offsets,inOffsets,sizeof(offsets));
            currentGPUBuffer = BUFFER_COUNT-1;
        }
        ~CMultiBufferedInterfaceBlock()
        {
            underlyingBuffer->drop();
        }
    public:
        static inline CMultiBufferedInterfaceBlock<BLOCK_STRUCT,BUFFER_COUNT>* createFromBuffer(IGPUBuffer* alreadyMadeBuffer, const size_t& firstOffset=0)
        {
            if (!alreadyMadeBuffer)
                return nullptr;

            if (firstOffset+sizeof(BLOCK_STRUCT)*BUFFER_COUNT>=alreadyMadeBuffer->getSize())
                return nullptr;

            size_t tmpOffsets[BUFFER_COUNT];
            for (size_t i=0; i<BUFFER_COUNT; i++)
            {
                tmpOffsets[i] = firstOffset+i*BUFFER_COUNT;
            }
            return new CMultiBufferedInterfaceBlock<BLOCK_STRUCT,BUFFER_COUNT>(alreadyMadeBuffer,tmpOffsets);
        }

        static inline CMultiBufferedInterfaceBlock<BLOCK_STRUCT,BUFFER_COUNT>* createFromBuffer(IGPUBuffer* alreadyMadeBuffer, const size_t inOffsets[BUFFER_COUNT])
        {
            if (!alreadyMadeBuffer)
                return nullptr;

            for (size_t i=0; i<BUFFER_COUNT; i++)
            {
                if (inOffsets[i]>=alreadyMadeBuffer->getSize())
                    return nullptr;
            }

            return new CMultiBufferedInterfaceBlock<BLOCK_STRUCT,BUFFER_COUNT>(alreadyMadeBuffer,inOffsets);
        }

        //! only creates the GPU local memory variant, if you want something more advanced, then make the buffer yourself and feed it to createFromBuffer
        static inline CMultiBufferedInterfaceBlock<BLOCK_STRUCT,BUFFER_COUNT>* create(IVideoDriver* driver)
        {
            IGPUBuffer* tmpBuffer = driver->createGPUBuffer(sizeof(BLOCK_STRUCT)*BUFFER_COUNT,nullptr,true);
            auto retval =  createFromBuffer(tmpBuffer);
            tmpBuffer->drop();
        }


        //! Modify data here, be sure to call swap() so GPU sees
        inline BLOCK_STRUCT& getBackBuffer() {return interfaceBackBuffer;}

        //! Extra parameters to prevent overwrite when you are using the end of the Struct for GPGPU writes
        //! `nextOffFence` needs to be a fence associated with the last use of the next buffer (i.e. from BUFFER_COUNT-1 swaps ago)
        inline void swap(const size_t& flushRangeStart=0, const size_t& flushRangeEndInclusive=sizeof(BLOCK_STRUCT)-1u, const bool& copyOverOldData=false, video::IDriverFence* nextOffFence=NULL)
        {
#ifdef _DEBUG
            assert(flushRangeStart<sizeof(BLOCK_STRUCT));
            assert(flushRangeEndInclusive<sizeof(BLOCK_STRUCT));
            assert(flushRangeStart<=flushRangeEndInclusive);
#endif // _DEBUG
            size_t prevBufferOff = currentGPUBuffer;
            size_t nextBufferOff = (prevBufferOff+1)%BUFFER_COUNT;

            size_t flushRangeEndExclusive = flushRangeEndInclusive+1u;

            // don't overwrite a buffer in-flight, make GPU wait
            if (nextOffFence)
                nextOffFence->waitGPU();

            if (copyOverOldData)
                copyOverHelperFunc();

            if (flushRangeEndInclusive>flushRangeStart)
            {
                uint8_t* ptr = &interfaceBackBuffer;
                ptr += flushRangeStart;
                underlyingBuffer->updateSubRange(nextBufferOff+flushRangeStart,flushRangeEndExclusive-flushRangeStart,ptr);
            }
        }

        inline size_t getCurrentBindingOffset() const {return offsets[currentGPUBuffer];}


        inline const IGPUBuffer* getUnderlyingBuffer() const {return underlyingBuffer;}

        inline const size_t* getOffsets() const {return offsets;}
    protected:
        IVideoDriver* m_driver;

        IGPUBuffer* underlyingBuffer;
        size_t offsets[BUFFER_COUNT];

        size_t currentGPUBuffer;

        BLOCK_STRUCT interfaceBackBuffer;


        inline void copyOverHelperFunc(const size_t& flushRangeStart, const size_t& flushRangeEndExclusive, const size_t& prevBufferOff, const size_t& nextBufferOff)
        {
            if (BUFFER_COUNT<2)
                return;

            size_t copyStart = flushRangeStart ? 0u:flushRangeEndExclusive;
            if (copyStart==sizeof(BLOCK_STRUCT)) //everything got overwritten
                return;

            //! TODO: For very large BLOCK_STRUCT (>8mb but needs benchmark) we should really do two copies if flushed region lies in the middle
            size_t copyEnd = flushRangeEndExclusive==sizeof(BLOCK_STRUCT) ? flushRangeStart:sizeof(BLOCK_STRUCT);
            if (copyEnd>copyStart)
                m_driver->bufferCopy(underlyingBuffer,underlyingBuffer,prevBufferOff+copyStart,nextBufferOff+copyStart,copyEnd-copyStart);
        }
};

//! TODO: Persistent Coherent and Persistent Flushed versions needed.
/* Notes:
Persistently Mapped Buffer version will need CPU waiting fence on swap
    Because we can't track what ranges have been written in a coherent buffer, swap will have to memcpy all of the previous scratch area onto the new one
    For the flushed variant, we memcpy the whole range anyway because the next scratch area is BUFFER_COUNT flushes behind
    UNLESS we create a history of flush ranges over the last BUFFER_COUNT swaps and min-max them to cull our memcpy (only useful for large structs, 256kB+)
*/

} // end namespace scene
} // end namespace irr

#endif


