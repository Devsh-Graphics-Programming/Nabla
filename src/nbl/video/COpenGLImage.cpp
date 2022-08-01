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
	GLsizei samples = m_creationParams.samples;
	switch (m_creationParams.type) // TODO what about multisample targets?
	{
	case IGPUImage::ET_1D:
		gl->extGlTextureStorage2D(
			name, target, m_creationParams.mipLevels, internalFormat,
			m_creationParams.extent.width, m_creationParams.arrayLayers
		);
		break;
	case IGPUImage::ET_2D:
		if (samples == 1)
			gl->extGlTextureStorage3D(
				name, target, m_creationParams.mipLevels, internalFormat,
				m_creationParams.extent.width, m_creationParams.extent.height, m_creationParams.arrayLayers
			);
		else
			gl->extGlTextureStorage3DMultisample(
				name, target, samples, internalFormat,
				m_creationParams.extent.width, m_creationParams.extent.height, m_creationParams.arrayLayers, GL_TRUE
			);
		break;
	case IGPUImage::ET_3D:
		gl->extGlTextureStorage3D(
			name, target, m_creationParams.mipLevels, internalFormat,
			m_creationParams.extent.width, m_creationParams.extent.height, m_creationParams.extent.depth
		);
		break;
	default:
		assert(false);
		break;
	}
	return true;
}

COpenGLImage::~COpenGLImage()
{
    preDestroyStep();

    auto* device = static_cast<IOpenGL_LogicalDevice*>(const_cast<ILogicalDevice*>(getOriginDevice()));
    // temporary fbos are created in the background to perform blits and color clears
    COpenGLFramebuffer::hash_t fbohash;
    if (asset::isDepthOrStencilFormat(m_creationParams.format))
        fbohash = COpenGLFramebuffer::getHashDepthStencilImage(this);
    else
        fbohash = COpenGLFramebuffer::getHashColorImage(this);
    device->destroyFramebuffer(fbohash);
    // destroy only if not observing (we own)
    if (!m_cachedCreationParams.skipHandleDestroy)
        device->destroyTexture(name);
}

void COpenGLImage::setObjectDebugName(const char* label) const
{
    IBackendObject::setObjectDebugName(label);

    auto* device = static_cast<IOpenGL_LogicalDevice*>(const_cast<ILogicalDevice*>(getOriginDevice()));
    device->setObjectDebugName(GL_TEXTURE, name, strlen(getObjectDebugName()), getObjectDebugName());
}

}