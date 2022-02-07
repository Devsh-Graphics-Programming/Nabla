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
    ESB_FRONT_LEFT = 0,
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

//! Legacy and deprecated system
class IVideoDriver : public IDriver
{
public:
    IVideoDriver(IrrlichtDevice* _dev)
        : IDriver(_dev) {}

    virtual bool initAuxContext() = 0;
    virtual bool deinitAuxContext() = 0;

    virtual bool bindGraphicsPipeline(const video::IGPURenderpassIndependentPipeline* _gpipeline) = 0;

    virtual bool bindComputePipeline(const video::IGPUComputePipeline* _cpipeline) = 0;

    virtual bool bindDescriptorSets(E_PIPELINE_BIND_POINT _pipelineType, const IGPUPipelineLayout* _layout,
        uint32_t _first, uint32_t _count, const IGPUDescriptorSet* const* _descSets, core::smart_refctd_dynamic_array<uint32_t>* _dynamicOffsets) = 0;

    virtual bool dispatch(uint32_t _groupCountX, uint32_t _groupCountY, uint32_t _groupCountZ) = 0;
    virtual bool dispatchIndirect(const IGPUBuffer* _indirectBuf, size_t _offset) = 0;

    virtual bool pushConstants(const IGPUPipelineLayout* _layout, uint32_t _stages, uint32_t _offset, uint32_t _size, const void* _values) = 0;

    //! Applications must call this method before performing any rendering.
    /** This method can clear the back- and the z-buffer.
		\param backBuffer Specifies if the back buffer should be
		cleared, which means that the screen is filled with the color
		specified. If this parameter is false, the back buffer will
		not be cleared and the color parameter is ignored.
		\param zBuffer Specifies if the depth buffer (z buffer) should
		be cleared. It is not nesesarry to do so if only 2d drawing is
		used.
		\param color The color used for back buffer clearing
		\param videoData Handle of another window, if you want the
		bitmap to be displayed on another window. If this is an empty
		element, everything will be displayed in the default window.
		Note: This feature is not fully implemented for all devices.
		\param sourceRect Pointer to a rectangle defining the source
		rectangle of the area to be presented. Set to null to present
		everything. Note: not implemented in all devices.
		\return False if failed. */
    virtual bool beginScene(bool backBuffer = true, bool zBuffer = true,
        SColor color = SColor(255, 0, 0, 0),
        const SExposedVideoData& videoData = SExposedVideoData(),
        core::rect<int32_t>* sourceRect = 0) = 0;

    //! Presents the rendered image to the screen.
    /** Applications must call this method after performing any
		rendering.
		\return False if failed and true if succeeded. */
    virtual bool endScene() = 0;

    //!
    virtual void issueGPUTextureBarrier() = 0;

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
        bool copyDepth = true, bool copyStencil = true,
        core::recti srcRect = core::recti(0, 0, 0, 0),
        core::recti dstRect = core::recti(0, 0, 0, 0),
        bool bilinearFilter = false) {}

    //!
    virtual void removeFrameBuffer(IFrameBuffer* framebuf) {}

    //! This only removes all IFrameBuffers created in the calling thread.
    virtual void removeAllFrameBuffers() {}

    //! Queries
    virtual void beginQuery(IQueryObject* query) = 0;
    virtual void endQuery(IQueryObject* query) = 0;

    //! Sets new multiple render targets.
    virtual bool setRenderTarget(IFrameBuffer* frameBuffer, bool setNewViewport = true) { return false; }

    //! Clears the ZBuffer.
    /** Note that you usually need not to call this method, as it
		is automatically done in IVideoDriver::beginScene() or
		IVideoDriver::setRenderTarget() if you enable zBuffer. But if
		you have to render some special things, you can clear the
		zbuffer during the rendering process with this method any time.
		*/
    virtual void clearZBuffer(const float& depth = 0.0) {}

    virtual void clearStencilBuffer(const int32_t& stencil) {}

    virtual void clearZStencilBuffers(const float& depth, const int32_t& stencil) {}

    virtual void clearColorBuffer(const E_FBO_ATTACHMENT_POINT& attachment, const int32_t* vals) {}
    virtual void clearColorBuffer(const E_FBO_ATTACHMENT_POINT& attachment, const uint32_t* vals) {}
    virtual void clearColorBuffer(const E_FBO_ATTACHMENT_POINT& attachment, const float* vals) {}

    virtual void clearScreen(const E_SCREEN_BUFFERS& buffer, const float* vals) {}
    virtual void clearScreen(const E_SCREEN_BUFFERS& buffer, const uint32_t* vals) {}

    //! Sets a new viewport.
    /** Every rendering operation is done into this new area.
		\param area: Rectangle defining the new area of rendering
		operations. */
    virtual void setViewPort(const core::rect<int32_t>& area) {}

    //! Gets the area of the current viewport.
    /** \return Rectangle of the current viewport. */
    virtual const core::rect<int32_t>& getViewPort() const = 0;

    //! Draws a mesh buffer
    /** \param mb Buffer to draw */
    virtual void drawMeshBuffer(const video::IGPUMeshBuffer* mb) = 0;

    //! Indirect Draw
    inline void drawArraysIndirect(const asset::SBufferBinding<IGPUBuffer> _vtxBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT],
        asset::E_PRIMITIVE_TOPOLOGY mode,
        const IGPUBuffer* indirectDrawBuff,
        size_t offset, size_t maxCount, size_t stride,
        const IGPUBuffer* countBuffer = nullptr, size_t countOffset = 0u)
    {
        return drawArraysIndirect(reinterpret_cast<const asset::SBufferBinding<const IGPUBuffer>*>(_vtxBindings), mode, indirectDrawBuff, offset, maxCount, stride, countBuffer, countOffset);
    }
    virtual void drawArraysIndirect(const asset::SBufferBinding<const IGPUBuffer> _vtxBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT],
        asset::E_PRIMITIVE_TOPOLOGY mode,
        const IGPUBuffer* indirectDrawBuff,
        size_t offset, size_t maxCount, size_t stride,
        const IGPUBuffer* countBuffer = nullptr, size_t countOffset = 0u) = 0;

    inline void drawIndexedIndirect(const asset::SBufferBinding<IGPUBuffer> _vtxBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT],
        asset::E_PRIMITIVE_TOPOLOGY mode,
        asset::E_INDEX_TYPE indexType, const IGPUBuffer* indexBuff,
        const IGPUBuffer* indirectDrawBuff,
        size_t offset, size_t maxCount, size_t stride,
        const IGPUBuffer* countBuffer = nullptr, size_t countOffset = 0u)
    {
        return drawIndexedIndirect(reinterpret_cast<const asset::SBufferBinding<const IGPUBuffer>*>(_vtxBindings), mode, indexType, indexBuff, indirectDrawBuff, offset, maxCount, stride, countBuffer, countOffset);
    }
    virtual void drawIndexedIndirect(const asset::SBufferBinding<const IGPUBuffer> _vtxBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT],
        asset::E_PRIMITIVE_TOPOLOGY mode,
        asset::E_INDEX_TYPE indexType, const IGPUBuffer* indexBuff,
        const IGPUBuffer* indirectDrawBuff,
        size_t offset, size_t maxCount, size_t stride,
        const IGPUBuffer* countBuffer = nullptr, size_t countOffset = 0u) = 0;

    //! Get the size of the screen or render window.
    /** \return Size of screen or render window. */
    virtual const core::dimension2d<uint32_t>& getScreenSize() const = 0;

    //! Get the size of the current render target
    /** This method will return the screen size if the driver
		doesn't support render to texture, or if the current render
		target is the screen.
		\return Size of render target or screen/window */
    virtual const core::dimension2d<uint32_t>& getCurrentRenderTargetSize() const = 0;

    //! Returns current frames per second value.
    /** This value is updated approximately every 1.5 seconds and
		is only intended to provide a rough guide to the average frame
		rate. It is not suitable for use in performing timing
		calculations or framerate independent movement.
		\return Approximate amount of frames per second drawn. */
    virtual int32_t getFPS() const = 0;

    //! Returns amount of primitives (mostly triangles) which were drawn in the last frame.
    /** Together with getFPS() very useful method for statistics.
		\param mode Defines if the primitives drawn are accumulated or
		counted per frame.
		\return Amount of primitives drawn in the last frame. */
    virtual uint32_t getPrimitiveCountDrawn(uint32_t mode = 0) const = 0;

    //! Event handler for resize events. Only used by the engine internally.
    /** Used to notify the driver that the window was resized.
		Usually, there is no need to call this method. */
    virtual void OnResize(const core::dimension2d<uint32_t>& size) = 0;

    //! Returns driver and operating system specific data about the IVideoDriver.
    /** This method should only be used if the engine should be
		extended without having to modify the source of the engine.
		\return Collection of device dependent pointers. */
    virtual const SExposedVideoData& getExposedVideoData() = 0;

    //! Enable or disable a clipping plane.
    /** There are at least 6 clipping planes available for the user
		to set at will.
		\param index The plane index. Must be between 0 and
		MaxUserClipPlanes.
		\param enable If true, enable the clipping plane else disable
		it. */
    virtual void enableClipPlane(uint32_t index, bool enable) {}
};

}  // end namespace video
}  // end namespace nbl

#endif
