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

#if false

namespace irr
{


namespace video
{

//! This is a WRITE-ONLY class!
template<class BLOCK_STRUCT, class IGPUBufferTYPE, size_t BUFFER_COUNT>
class CMultiBufferedInterfaceBlockBase : public virtual IReferenceCounted, public Interface //maybe make non ref-counted in the future
{
    public:
        typedef BLOCK_STRUCT InterfaceBlockType;

        //! Modify data here, be sure to call swap() after all the writing so GPU sees the changes
        virtual BLOCK_STRUCT& getBackBuffer() = 0;

        //! Extra parameters to prevent overwrite when you are using the end of the Struct for GPGPU writes
        //! `copyOverOldData` means whether you'll need to rewrite the entire backbuffer entirely to get sensible results
        //! If using `nextOffFence` then it needs to be a fence placed after the use of the buffer at offset `getCurrentBindingOffset()` from at most BUFFER_COUNT-1 (copyOverOldData=false) or BUFFER_COUNT-2 (copyOverOldData=true) swaps ago
        inline bool swap(const size_t& flushRangeStart=0, const size_t& flushRangeEndInclusive=sizeof(BLOCK_STRUCT)-1u, const bool& copyOverOldData=false, video::IDriverFence* nextOffFence=NULL)
        {
#ifdef _DEBUG
            assert(flushRangeStart<sizeof(BLOCK_STRUCT));
            assert(flushRangeEndInclusive<sizeof(BLOCK_STRUCT));
            assert(flushRangeStart<=flushRangeEndInclusive);
            assert(!nextOffFence || BUFFER_COUNT>1);
#endif // _DEBUG
            size_t nextSubBuffer = (currentSubBuffer+1)%BUFFER_COUNT;
            size_t nextBufferOff = offsets[nextSubBuffer];

            size_t flushRangeEndExclusive = flushRangeEndInclusive+1u;

            if (this->internalSwap(flushRangeStart,flushRangeEndExclusive,copyOverOldData,nextBufferOff,nextOffFence))
            {
                currentSubBuffer = nextSubBuffer;
                return true;
            }

            return false;
        }

        inline size_t getCurrentBindingOffset() const {return offsets[currentSubBuffer];}

        inline const IGPUBuffer* getUnderlyingBuffer() const {return underlyingBuffer;}

        inline const size_t* getOffsets() const {return offsets;}


    protected:
        CMultiBufferedInterfaceBlockBase(IGPUBufferTYPE* inBuffer, const size_t inOffsets[BUFFER_COUNT]) : underlyingBuffer(inBuffer)
        {
            static_assert(BUFFER_COUNT>0,"What on earth are you doing? Zero Buffering is not acceptable!");
            underlyingBuffer->grab();
            memcpy(offsets,inOffsets,sizeof(offsets));
            currentSubBuffer = BUFFER_COUNT-1;
        }
        virtual ~CMultiBufferedInterfaceBlockBase()
        {
            underlyingBuffer->drop();
        }

        //
        virtual bool internalSwap(const size_t& flushRangeStart, const size_t& flushRangeEndExclusive, const bool& copyOverOldData, const size_t& nextBufferOff, video::IDriverFence* nextOffFence=NULL) = 0;


        IGPUBufferTYPE* underlyingBuffer;

        size_t offsets[BUFFER_COUNT];

        size_t currentSubBuffer;
};


//!
template<class BLOCK_STRUCT, class IGPUBufferTYPE, size_t BUFFER_COUNT> class CMultiBufferedInterfaceBlock : public CMultiBufferedInterfaceBlockBase<BLOCK_STRUCT,IGPUBufferTYPE,BUFFER_COUNT> {};

//
template<class BLOCK_STRUCT, size_t BUFFER_COUNT>
class CMultiBufferedInterfaceBlock<BLOCK_STRUCT,IGPUBuffer,BUFFER_COUNT> : public CMultiBufferedInterfaceBlockBase<BLOCK_STRUCT,IGPUBuffer,BUFFER_COUNT>
{
        typedef CMultiBufferedInterfaceBlockBase<BLOCK_STRUCT,IGPUBuffer,BUFFER_COUNT> specBaseType;

    protected:
        CMultiBufferedInterfaceBlock(IGPUBuffer* inBuffer, const size_t inOffsets[BUFFER_COUNT], video::IVideoDriver* inDriver)
            : CMultiBufferedInterfaceBlockBase<BLOCK_STRUCT,IGPUBuffer,BUFFER_COUNT>(inBuffer,inOffsets), m_driver(inDriver) {}

    public:
        static inline CMultiBufferedInterfaceBlock<BLOCK_STRUCT,IGPUBuffer,BUFFER_COUNT>* createFromBuffer(video::IVideoDriver* driver, IGPUBuffer* alreadyMadeBuffer, const size_t& firstOffset=0)
        {
            if (!alreadyMadeBuffer||!alreadyMadeBuffer->canUpdateSubRange())
                return nullptr;

            if (firstOffset+sizeof(BLOCK_STRUCT)*BUFFER_COUNT>alreadyMadeBuffer->getSize())
                return nullptr;

            size_t tmpOffsets[BUFFER_COUNT];
            for (size_t i=0; i<BUFFER_COUNT; i++)
            {
                tmpOffsets[i] = firstOffset+i*BUFFER_COUNT;
            }
            return new CMultiBufferedInterfaceBlock<BLOCK_STRUCT,IGPUBuffer,BUFFER_COUNT>(alreadyMadeBuffer,tmpOffsets,driver);
        }

        static inline CMultiBufferedInterfaceBlock<BLOCK_STRUCT,IGPUBuffer,BUFFER_COUNT>* createFromBuffer(video::IVideoDriver* driver, IGPUBuffer* alreadyMadeBuffer, const size_t inOffsets[BUFFER_COUNT])
        {
            if (!alreadyMadeBuffer||!alreadyMadeBuffer->canUpdateSubRange())
                return nullptr;

            for (size_t i=0; i<BUFFER_COUNT; i++)
            {
                if (inOffsets[i]+sizeof(BLOCK_STRUCT)>alreadyMadeBuffer->getSize())
                    return nullptr;
            }

            return new CMultiBufferedInterfaceBlock<BLOCK_STRUCT,IGPUBuffer,BUFFER_COUNT>(alreadyMadeBuffer,inOffsets,driver);
        }

        //! only creates the GPU local memory variant, if you want something more advanced, then make the buffer yourself and feed it to createFromBuffer
        static inline CMultiBufferedInterfaceBlock<BLOCK_STRUCT,IGPUBuffer,BUFFER_COUNT>* create(IVideoDriver* driver)
        {
            IGPUBuffer* tmpBuffer = driver->createGPUBuffer(sizeof(BLOCK_STRUCT)*BUFFER_COUNT,nullptr,true);
            auto retval =  createFromBuffer(driver,tmpBuffer);
            tmpBuffer->drop();
            return retval;
        }


        virtual BLOCK_STRUCT& getBackBuffer() {return interfaceBackBuffer;}
    protected:
        IVideoDriver* m_driver;

        BLOCK_STRUCT interfaceBackBuffer;


        virtual bool internalSwap(const size_t& flushRangeStart, const size_t& flushRangeEndExclusive, const bool& copyOverOldData, const size_t& nextBufferOff, video::IDriverFence* nextOffFence=NULL)
        {
            // don't overwrite a buffer in-flight, make GPU wait
            if (nextOffFence)
                nextOffFence->waitGPU();

            if (flushRangeEndExclusive>flushRangeStart)
            {
                uint8_t* ptr = reinterpret_cast<uint8_t*>(&interfaceBackBuffer);
                ptr += flushRangeStart;
                specBaseType::underlyingBuffer->updateSubRange(nextBufferOff+flushRangeStart,flushRangeEndExclusive-flushRangeStart,ptr);
            }

            if (copyOverOldData && BUFFER_COUNT>=2)
            {
                size_t nextNextBufferOff = specBaseType::offsets[(specBaseType::currentSubBuffer+2)%BUFFER_COUNT];
                m_driver->copyBuffer(specBaseType::underlyingBuffer,specBaseType::underlyingBuffer,nextBufferOff,nextNextBufferOff,sizeof(BLOCK_STRUCT));
            }

            return true;
        }
};

template<class BLOCK_STRUCT, size_t BUFFER_COUNT>
class CMultiBufferedInterfaceBlock<BLOCK_STRUCT,IGPUMappedBuffer,BUFFER_COUNT> : public CMultiBufferedInterfaceBlockBase<BLOCK_STRUCT,IGPUMappedBuffer,BUFFER_COUNT>
{
        typedef CMultiBufferedInterfaceBlockBase<BLOCK_STRUCT,IGPUMappedBuffer,BUFFER_COUNT> specBaseType;

    protected:
        CMultiBufferedInterfaceBlock(IGPUMappedBuffer* inBuffer, const size_t inOffsets[BUFFER_COUNT])
            : CMultiBufferedInterfaceBlockBase<BLOCK_STRUCT,IGPUMappedBuffer,BUFFER_COUNT>(inBuffer,inOffsets)
        {
            /*
            static_assert(BUFFER_COUNT>=4,
                "You need at least 3 ranges; one for GPU to read from, one to write to, and one to separate the range GPU reads from the one you are writing to while you're swapping.\
                If you use a fence to wait then you can get away with 1 or 2.\
                But GPU buffers frames in advance, so you actually need 4+ .");
            */
        }

    public:
        static inline CMultiBufferedInterfaceBlock<BLOCK_STRUCT,IGPUMappedBuffer,BUFFER_COUNT>* createFromBuffer(IGPUMappedBuffer* alreadyMadeBuffer, const size_t& firstOffset=0)
        {
            if (!alreadyMadeBuffer)
                return nullptr;
#ifdef _DEBUG
            if (!alreadyMadeBuffer->isMappedBuffer())
                return nullptr;
#endif // _DEBUG
            if (firstOffset+sizeof(BLOCK_STRUCT)*BUFFER_COUNT>alreadyMadeBuffer->getSize())
                return nullptr;

            size_t tmpOffsets[BUFFER_COUNT];
            for (size_t i=0; i<BUFFER_COUNT; i++)
            {
                tmpOffsets[i] = firstOffset+i*BUFFER_COUNT;
            }
            return new CMultiBufferedInterfaceBlock<BLOCK_STRUCT,IGPUMappedBuffer,BUFFER_COUNT>(alreadyMadeBuffer,tmpOffsets);
        }

        static inline CMultiBufferedInterfaceBlock<BLOCK_STRUCT,IGPUMappedBuffer,BUFFER_COUNT>* createFromBuffer(IGPUMappedBuffer* alreadyMadeBuffer, const size_t inOffsets[BUFFER_COUNT])
        {
            if (!alreadyMadeBuffer)
                return nullptr;

            for (size_t i=0; i<BUFFER_COUNT; i++)
            {
                if (inOffsets[i]+sizeof(BLOCK_STRUCT)>alreadyMadeBuffer->getSize())
                    return nullptr;
            }

            return new CMultiBufferedInterfaceBlock<BLOCK_STRUCT,IGPUMappedBuffer,BUFFER_COUNT>(alreadyMadeBuffer,inOffsets);
        }

        static inline CMultiBufferedInterfaceBlock<BLOCK_STRUCT,IGPUMappedBuffer,BUFFER_COUNT>* create(IVideoDriver* driver, const E_GPU_BUFFER_ACCESS &usagePattern=EGBA_WRITE, const bool& inCPUMem=true)
        {
#ifdef _DEBUG
            if (usagePattern==EGBA_READ||usagePattern==EGBA_NONE)
                return nullptr;
#endif // _DEBUG

            IGPUMappedBuffer* tmpBuffer = driver->createPersistentlyMappedBuffer(sizeof(BLOCK_STRUCT)*BUFFER_COUNT,nullptr,usagePattern,true,inCPUMem);
            auto retval =  createFromBuffer(tmpBuffer);
            tmpBuffer->drop();
            return retval;
        }


        virtual BLOCK_STRUCT& getBackBuffer()
        {
            size_t nextSubBuff = (specBaseType::currentSubBuffer+1)%BUFFER_COUNT;

            uint8_t* dstPtr = reinterpret_cast<uint8_t*>(specBaseType::underlyingBuffer->getPointer());
            dstPtr += specBaseType::offsets[nextSubBuff];
            return *reinterpret_cast<BLOCK_STRUCT*>(dstPtr);
        }
    protected:
        // If you don't want the buffer contents to be preserved, then don't flush
        virtual bool internalSwap(const size_t& flushRangeStart, const size_t& flushRangeEndExclusive, const bool& copyOverOldData, const size_t& nextBufferOff, video::IDriverFence* nextOffFence=NULL)
        {
            auto buff = specBaseType::underlyingBuffer;
            uint8_t* basePtr = reinterpret_cast<uint8_t*>(buff->getPointer());

            // don't overwrite a buffer in-flight, make GPU wait
            if (nextOffFence)
            {
                switch (nextOffFence->waitCPU(1000000000u))
                {
                    case EDFR_CONDITION_SATISFIED:
                    case EDFR_ALREADY_SIGNALED:
                        return true;
                        break;
                    default:
                        return false;
                        break;
                }
            }

            if (copyOverOldData)
            {
                size_t nextNextSubBuffer = (specBaseType::currentSubBuffer+2)%BUFFER_COUNT;
                memcpy(basePtr+nextNextSubBuffer,basePtr+nextBufferOff,sizeof(BLOCK_STRUCT));
            }

            //! Flush Dest range on the flushable buffer specialization, it should actually be the other way round
            //if (flushRangeEndExclusive>flushRangeStart)
            //    underlyingBuffer->flush(nextBufferOff+flushRangeStart,nextBufferOff+flushRangeEndExclusive);

            return true;
        }
};

//! TODO: Different Persistent Coherent and Persistent Flushed versions needed.

} // end namespace scene
} // end namespace irr

#endif // false

#endif


