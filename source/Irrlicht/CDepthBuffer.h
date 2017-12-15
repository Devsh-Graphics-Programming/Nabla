// Copyright (C) 2002-2012 Nikolaus Gebhardt / Thomas Alten
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_Z_BUFFER_H_INCLUDED__
#define __C_Z_BUFFER_H_INCLUDED__

#include "IDepthBuffer.h"

namespace irr
{
namespace video
{

	class CDepthBuffer : public IDepthBuffer
	{
        protected:
            //! destructor
            virtual ~CDepthBuffer();

        public:
            //! constructor
            CDepthBuffer(const core::dimension2d<uint32_t>& size);

            //! clears the zbuffer
            virtual void clear();

            //! sets the new size of the zbuffer
            virtual void setSize(const core::dimension2d<uint32_t>& size);

            //! returns the size of the zbuffer
            virtual const core::dimension2d<uint32_t>& getSize() const;

            //! locks the zbuffer
            virtual void* lock() { return (void*) Buffer; }

            //! unlocks the zbuffer
            virtual void unlock() {}

            //! returns pitch of depthbuffer (in bytes)
            virtual uint32_t getPitch() const { return Pitch; }


        private:

            uint8_t* Buffer;
            core::dimension2d<uint32_t> Size;
            uint32_t TotalSize;
            uint32_t Pitch;
	};


	class CStencilBuffer : public IStencilBuffer
	{
        protected:
            //! destructor
            virtual ~CStencilBuffer();

        public:
            //! constructor
            CStencilBuffer(const core::dimension2d<uint32_t>& size);

            //! clears the zbuffer
            virtual void clear();

            //! sets the new size of the zbuffer
            virtual void setSize(const core::dimension2d<uint32_t>& size);

            //! returns the size of the zbuffer
            virtual const core::dimension2d<uint32_t>& getSize() const;

            //! locks the zbuffer
            virtual void* lock() { return (void*) Buffer; }

            //! unlocks the zbuffer
            virtual void unlock() {}

            //! returns pitch of depthbuffer (in bytes)
            virtual uint32_t getPitch() const { return Pitch; }


        private:

            uint8_t* Buffer;
            core::dimension2d<uint32_t> Size;
            uint32_t TotalSize;
            uint32_t Pitch;
	};

} // end namespace video
} // end namespace irr

#endif

