#ifndef _IRR_EXT_SCREEN_SHOT_INCLUDED_
#define _IRR_EXT_SCREEN_SHOT_INCLUDED_

#include "irrlicht.h"

#include "../source/Irrlicht/COpenGLBuffer.h"
#include "../source/Irrlicht/COpenGLExtensionHandler.h"

namespace irr
{
namespace ext
{
namespace ScreenShot
{

core::smart_refctd_ptr<video::IDriverFence> createScreenShot(video::IDriver* driver, video::ITexture* source, video::IGPUBuffer* destination, uint32_t sourceMipLevel=0u, size_t destOffset=0ull, bool implicitflush=true)
{
	// will change this, https://github.com/buildaworldnet/IrrlichtBAW/issues/148
	if (isBlockCompressionFormat(source->getColorFormat()))
		return nullptr;

	auto gltex = dynamic_cast<video::COpenGLTexture*>(source);
	GLenum colorformat=GL_INVALID_ENUM, type=GL_INVALID_ENUM;
	video::COpenGLTexture::getOpenGLFormatAndParametersFromColorFormat(source->getColorFormat(),colorformat,type);

	video::COpenGLExtensionHandler::extGlBindBuffer(GL_PIXEL_PACK_BUFFER, static_cast<video::COpenGLBuffer*>(destination)->getOpenGLName());
	video::COpenGLExtensionHandler::extGlGetTextureImage(	gltex->getOpenGLName(),gltex->getOpenGLTextureType(),sourceMipLevel,
															colorformat,type,source->getPitch()*source->getSize()[1],reinterpret_cast<void*>(destOffset));
	video::COpenGLExtensionHandler::extGlBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

	return driver->placeFence(implicitflush);
}

} // namespace ScreenShot
} // namespace ext
} // namespace irr

#endif // _IRR_EXT_SCREEN_SHOT_INCLUDED_
