// Copyright (C) 2016 Mateusz "DeVsh" Kielan
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __I_CPU_BUFFER_H_INCLUDED__
#define __I_CPU_BUFFER_H_INCLUDED__

#include "irr/core/IBuffer.h"
#include "IAsset.h"

namespace irr
{
namespace core
{

class ICPUBuffer : public IBuffer, public asset::IAsset
{
    protected:
        virtual ~ICPUBuffer()
        {
            if (data)
                _IRR_ALIGNED_FREE(data);
        }
    public:
		//! Constructor.
		/** @param sizeInBytes Size in bytes. If `dat` argument is present, it denotes size of data pointed by `dat`, otherwise - size of data to be allocated.
		@param dat Optional parameter. Pointer to data, must be allocated with `_IRR_ALIGNED_MALLOC`. Note that pointed data will not be copied to some internal buffer storage, but buffer will operate on original data pointed by `dat`.
		*/
        ICPUBuffer(const size_t &sizeInBytes, void *dat = NULL) : size(0), data(dat)
        {
			if (!data)
				data = _IRR_ALIGNED_MALLOC(sizeInBytes,_IRR_SIMD_ALIGNMENT);
            if (!data)
                return;

            size = sizeInBytes;
        }

        virtual void convertToDummyObject() override 
        {
            _IRR_ALIGNED_FREE(data);
            data = nullptr;
            size = 0ull;
            isDummyObjectForCacheAliasing = true;
        }
        virtual asset::IAsset::E_TYPE getAssetType() const override { return asset::IAsset::ET_BUFFER; }

        virtual size_t conservativeSizeEstimate() const override { return getSize(); }

        //! Returns size in bytes.
        virtual const uint64_t& getSize() const {return size;}
/*
		//! Reallocates internal data. Invalidate any sizes, pointers etc. returned before!
		/** @param newSize New size of memory.
		@param forceRetentionOfData Doesn't matter.
		@param reallocateIfShrink Whether to perform reallocation even if it means to shrink the buffer (lose some data).
		@returns True on success or false otherwise.
		*
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
*/
		//! Returns pointer to data.
        /** WARNING: RESIZE will invalidate pointer.
		*/
        virtual const void* getPointer() const {return data;}
		//! Returns pointer to data.
		/** WARNING: RESIZE will invalidate pointer.
		*/
        virtual void* getPointer() {return data;}

    private:
        uint64_t size;
        void* data;
};

} // end namespace scene
} // end namespace irr

#endif
