// Copyright (C) 2002-2012 Nikolaus Gebhardt / Thomas Alten
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "IrrCompileConfig.h"
#include "SoftwareDriver2_compile_config.h"
#include "CDepthBuffer.h"

#ifdef _IRR_COMPILE_WITH_BURNINGSVIDEO_

namespace irr
{
namespace video
{


//! constructor
CDepthBuffer::CDepthBuffer(const core::dimension2d<uint32_t>& size)
: Buffer(0), Size(0,0)
{
	#ifdef _IRR_DEBUG
	setDebugName("CDepthBuffer");
	#endif

	setSize(size);
}



//! destructor
CDepthBuffer::~CDepthBuffer()
{
	if (Buffer)
		delete [] Buffer;
}



//! clears the zbuffer
void CDepthBuffer::clear()
{

#ifdef SOFTWARE_DRIVER_2_USE_WBUFFER
	float zMax = 0.f;
#else
	float zMax = 1.f;
#endif

	uint32_t zMaxValue;
	zMaxValue = IR(zMax);

	memset32 ( Buffer, zMaxValue, TotalSize );
}



//! sets the new size of the zbuffer
void CDepthBuffer::setSize(const core::dimension2d<uint32_t>& size)
{
	if (size == Size)
		return;

	Size = size;

	if (Buffer)
		delete [] Buffer;

	Pitch = size.Width * sizeof ( fp24 );
	TotalSize = Pitch * size.Height;
	Buffer = new uint8_t[TotalSize];
	clear ();
}



//! returns the size of the zbuffer
const core::dimension2d<uint32_t>& CDepthBuffer::getSize() const
{
	return Size;
}

// -----------------------------------------------------------------

//! constructor
CStencilBuffer::CStencilBuffer(const core::dimension2d<uint32_t>& size)
: Buffer(0), Size(0,0)
{
	#ifdef _IRR_DEBUG
	setDebugName("CDepthBuffer");
	#endif

	setSize(size);
}



//! destructor
CStencilBuffer::~CStencilBuffer()
{
	if (Buffer)
		delete [] Buffer;
}



//! clears the zbuffer
void CStencilBuffer::clear()
{
	memset32 ( Buffer, 0, TotalSize );
}



//! sets the new size of the zbuffer
void CStencilBuffer::setSize(const core::dimension2d<uint32_t>& size)
{
	if (size == Size)
		return;

	Size = size;

	if (Buffer)
		delete [] Buffer;

	Pitch = size.Width * sizeof ( uint32_t );
	TotalSize = Pitch * size.Height;
	Buffer = new uint8_t[TotalSize];
	clear ();
}



//! returns the size of the zbuffer
const core::dimension2d<uint32_t>& CStencilBuffer::getSize() const
{
	return Size;
}



} // end namespace video
} // end namespace irr

#endif // _IRR_COMPILE_WITH_BURNINGSVIDEO_

namespace irr
{
namespace video
{

//! creates a ZBuffer
IDepthBuffer* createDepthBuffer(const core::dimension2d<uint32_t>& size)
{
	#ifdef _IRR_COMPILE_WITH_BURNINGSVIDEO_
	return new CDepthBuffer(size);
	#else
	return 0;
	#endif // _IRR_COMPILE_WITH_BURNINGSVIDEO_
}


//! creates a ZBuffer
IStencilBuffer* createStencilBuffer(const core::dimension2d<uint32_t>& size)
{
	#ifdef _IRR_COMPILE_WITH_BURNINGSVIDEO_
	return new CStencilBuffer(size);
	#else
	return 0;
	#endif // _IRR_COMPILE_WITH_BURNINGSVIDEO_
}

} // end namespace video
} // end namespace irr



