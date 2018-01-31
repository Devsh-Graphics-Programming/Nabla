// Copyright (C) 2016 Mateusz "DeVsh" Kielan
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __I_CPU_BUFFER_H_INCLUDED__
#define __I_CPU_BUFFER_H_INCLUDED__

#include "IBuffer.h"

namespace irr
{
namespace core
{

//! Persistently Mapped buffer
class ICPUBuffer : public IBuffer
{
    protected:
        virtual ~ICPUBuffer()
        {
            if (data)
                free(data);
        }
    public:
        ICPUBuffer(const size_t &sizeInBytes, void *dat = NULL) : size(0), data(dat)
        {
			if (!data)
				data = malloc(sizeInBytes);
            if (!data)
                return;

            size = sizeInBytes;
        }

        virtual E_BUFFER_TYPE getBufferType() const {return EBT_UNSPECIFIED_BUFFER;}
        //! size in BYTES
        virtual const uint64_t& getSize() const {return size;}

        //! returns true on success (some types always return false since they dont support resize)
        //! This function will invalidate any sizes, pointers etc. returned before!
        virtual bool reallocate(const size_t &newSize, const bool& forceRetentionOfData=false, const bool &reallocateIfShrink=false)
        {
            if (size==newSize)
                return true;

            if ((!reallocateIfShrink)&&size>newSize)
            {
                size = newSize;
                return true;
            }

            data = realloc(data,newSize);
            if (!data)
            {
                size = 0;
                return false;
            }

            size = newSize;
            return true;
        }

        //! WARNING: RESIZE will invalidate pointer
        virtual const void* getPointer() const {return data;}
        virtual void* getPointer() {return data;}

    private:
        uint64_t size;
        void* data;
};

} // end namespace scene
} // end namespace irr

#endif
