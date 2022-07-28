#include "nbl/video/COpenGLImage.h"
#include "nbl/video/IOpenGL_LogicalDevice.h"
#include "nbl/video/IOpenGL_FunctionTable.h"
#include "nbl/video/COpenGLFramebuffer.h"
#include "nbl/video/COpenGLCommon.h"

namespace nbl::video
{

bool COpenGLImage::initMemory(
	IOpenGL_FunctionTable* gl,
	core::bitflag<E_MEMORY_ALLOCATE_FLAGS> allocateFlags,
	core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS> memoryPropertyFlags)
{
	if (!IOpenGLMemoryAllocation::initMemory(gl, allocateFlags, memoryPropertyFlags))
		return false;
	GLsizei samples = params.samples;
	switch (params.type) // TODO what about multisample targets?
	{
	case IGPUImage::ET_1D:
		gl->extGlTextureStorage2D(name, target, params.mipLevels, internalFormat,
			params.extent.width, params.arrayLayers);
		break;
	case IGPUImage::ET_2D:
		if (samples == 1)
			gl->extGlTextureStorage3D(name, target, params.mipLevels, internalFormat, params.extent.width, params.extent.height, params.arrayLayers);
		else
			gl->extGlTextureStorage3DMultisample(name, target, samples, internalFormat, params.extent.width, params.extent.height, params.arrayLayers, GL_TRUE);
		break;
	case IGPUImage::ET_3D:
		gl->extGlTextureStorage3D(name, target, params.mipLevels, internalFormat,
			params.extent.width, params.extent.height, params.extent.depth);
		break;
	default:
		assert(false);
		break;
	}
}

COpenGLImage::~COpenGLImage()
{
	if (m_optionalBackingSwapchain)
	{
		freeSwapchainImageExists();
	}

	// TODO Deletion behaviour here for swapchain images is the same as would've happened inside the 
	// COpenGL_SwapchainThreadHandler quit, but won't happen in the same thread
    auto* device = static_cast<IOpenGL_LogicalDevice*>(const_cast<ILogicalDevice*>(getOriginDevice()));
    device->destroyTexture(name);
    // temporary fbos are created in the background to perform blits and color clears
    COpenGLFramebuffer::hash_t fbohash;
    if (asset::isDepthOrStencilFormat(params.format))
        fbohash = COpenGLFramebuffer::getHashDepthStencilImage(this);
    else
        fbohash = COpenGLFramebuffer::getHashColorImage(this);
    device->destroyFramebuffer(fbohash);
}

void COpenGLImage::setObjectDebugName(const char* label) const
{
    IBackendObject::setObjectDebugName(label);

    auto* device = static_cast<IOpenGL_LogicalDevice*>(const_cast<ILogicalDevice*>(getOriginDevice()));
    device->setObjectDebugName(GL_TEXTURE, name, strlen(getObjectDebugName()), getObjectDebugName());
}

}