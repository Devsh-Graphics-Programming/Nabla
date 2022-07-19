// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_I_VIDEO_DRIVER_H_INCLUDED__
#define __NBL_I_VIDEO_DRIVER_H_INCLUDED__


namespace nbl
{
namespace video
{
#if 0
	//! Legacy and deprecated system
	class NBL_API IVideoDriver : public IDriver
	{
	public:
		//!
		virtual void issueGPUTextureBarrier() =0;

		//! Event handler for resize events. Only used by the engine internally.
		/** Used to notify the driver that the window was resized.
		Usually, there is no need to call this method. */
		virtual void OnResize(const core::dimension2d<uint32_t>& size) =0;

	};
#endif

} // end namespace video
} // end namespace nbl


#endif
