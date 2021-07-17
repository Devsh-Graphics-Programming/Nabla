// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_I_VIDEO_DRIVER_H_INCLUDED__
#define __NBL_I_VIDEO_DRIVER_H_INCLUDED__

#include "rect.h"
#include "SColor.h"
#include "matrixutil.h"
#include "dimension2d.h"
#include "position2d.h"
#include "IDriverFence.h"
#include "SExposedVideoData.h"
#include "IDriver.h"
#include "nbl/video/IGPUBufferView.h"
#include "nbl/video/IGPURenderpassIndependentPipeline.h"

namespace nbl
{
namespace video
{
	enum E_SCREEN_BUFFERS
	{
		ESB_FRONT_LEFT=0,
		ESB_FRONT_RIGHT,
		ESB_BACK_LEFT,
		ESB_BACK_RIGHT
	};
	//TODO move to IGPUCommandBuffer.h or higher level header
    enum E_PIPELINE_BIND_POINT
    {
        EPBP_GRAPHICS = 0,
        EPBP_COMPUTE = 1,
        EPBP_COUNT
    };
#if 0
	//! Legacy and deprecated system
	class IVideoDriver : public IDriver
	{
	public:
        IVideoDriver(asset::IAssetManager* assmgr) : IDriver(assmgr) {}

		//!
		virtual void issueGPUTextureBarrier() =0;

		//! Allows data in one framebuffer to be blitted to another framebuffer
		/** 
			A blit operation is a special form of copy operation. It copies a
			rectangular area of pixels from one framebuffer to another. Note that
			you should take care of your attachement inputs, so if for instance
			their depth attachements don't match - you must not try to copy depth
			between them.

			\param in Specifies an in framebuffer which data will be copied to out framebuffer.
			\param out Specifies an out framebuffer that will be taking data from in framebuffer.
			\param copyDepth Specifies whether depth attachement should be copied.
			\param copyStencil Specifies whether stencil attachement should be copied.
			\param srcRect Rectangular area in pixels for original source needed to copy to \bdstRect\b.
			\param dstRect Rectangular area in pixels for destination source where \bsrcRect\b is a reference.

			It is perfectly valid to blit from or to the Default Framebuffer,
			in such a case use \bnullptr\b.
		*/

		virtual void blitRenderTargets(IFrameBuffer* in, IFrameBuffer* out,
                                        bool copyDepth=true, bool copyStencil=true,
										core::recti srcRect=core::recti(0,0,0,0),
										core::recti dstRect=core::recti(0,0,0,0),
										bool bilinearFilter=false) {}

		//! Queries
		virtual void beginQuery(IQueryObject* query) = 0;
		virtual void endQuery(IQueryObject* query) = 0;

		//! Sets a new viewport.
		/** Every rendering operation is done into this new area.
		\param area: Rectangle defining the new area of rendering
		operations. */
		virtual void setViewPort(const core::rect<int32_t>& area) {}

		//! Gets the area of the current viewport.
		/** \return Rectangle of the current viewport. */
		virtual const core::rect<int32_t>& getViewPort() const =0;

		//! Draws a mesh buffer
		/** \param mb Buffer to draw */
		virtual void drawMeshBuffer(const video::IGPUMeshBuffer* mb) =0;

		//! Get the size of the screen or render window.
		/** \return Size of screen or render window. */
		virtual const core::dimension2d<uint32_t>& getScreenSize() const =0;

		//! Event handler for resize events. Only used by the engine internally.
		/** Used to notify the driver that the window was resized.
		Usually, there is no need to call this method. */
		virtual void OnResize(const core::dimension2d<uint32_t>& size) =0;

	};
#endif

} // end namespace video
} // end namespace nbl


#endif
