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

// change this to produce a CImageData backed by a mapped GPU buffer
core::smart_refctd_ptr<video::IDriverFence> createScreenShot(video::IDriver* driver, core::rect<uint32_t> sourceRect, asset::E_FORMAT _outFormat, video::IGPUBuffer* destination, size_t destOffset=0ull, bool implicitflush=true)
{
	// will change this, https://github.com/buildaworldnet/IrrlichtBAW/issues/148
	if (isBlockCompressionFormat(_outFormat))
		return nullptr;

	GLenum colorformat=GL_INVALID_ENUM, type=GL_INVALID_ENUM;
	video::COpenGLTexture::getOpenGLFormatAndParametersFromColorFormat(_outFormat,colorformat,type);

	video::COpenGLExtensionHandler::extGlBindBuffer(GL_PIXEL_PACK_BUFFER, static_cast<video::COpenGLBuffer*>(destination)->getOpenGLName());
	glReadPixels(sourceRect.LowerRightCorner.X, sourceRect.LowerRightCorner.Y, sourceRect.getWidth(), sourceRect.getHeight(), colorformat, type, reinterpret_cast<void*>(destOffset));
	video::COpenGLExtensionHandler::extGlBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

	return driver->placeFence(implicitflush);
}
/*
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
*/

template<typename PathOrFile>
void writeBufferAsImageToFile(asset::IAssetManager* mgr, const PathOrFile& _outFile, core::vector2d<uint32_t> _size, asset::E_FORMAT _format, video::IGPUBuffer* buff, size_t offset=0ull)
{
	const uint32_t zero[3] = { 0,0,0 };
	const uint32_t sizeArray[3] = { _size.X,_size.Y,1u };
	auto img = new asset::CImageData(reinterpret_cast<uint8_t*>(buff->getBoundMemory()->getMappedPointer())+offset, zero, sizeArray, 0u, _format);

	asset::IAssetWriter::SAssetWriteParams wparams(img);
	mgr->writeAsset(_outFile, wparams);
	
	img->drop();
}


template<typename PathOrFile>
void dirtyCPUStallingScreenshot(IrrlichtDevice* device, const PathOrFile& _outFile, core::rect<uint32_t> sourceRect, asset::E_FORMAT _format)
{
	auto assetManager = device->getAssetManager();
	auto driver = device->getVideoDriver();

	auto buff = core::smart_refctd_ptr<video::IGPUBuffer>(driver->createDownStreamingGPUBufferOnDedMem((asset::getBytesPerPixel(_format) * sourceRect.getArea()).getIntegerApprox()), core::dont_grab); // TODO
	buff->getBoundMemory()->mapMemoryRange(video::IDriverMemoryAllocation::EMCAF_READ,{0u,buff->getSize()});

	auto fence = ext::ScreenShot::createScreenShot(driver, sourceRect, _format, buff.get());
	fence->waitCPU(10000000000000ull, true);
	ext::ScreenShot::writeBufferAsImageToFile(assetManager, "screenshot.png", {sourceRect.getWidth(),sourceRect.getHeight()}, _format, buff.get());
}


} // namespace ScreenShot
} // namespace ext
} // namespace irr

#endif // _IRR_EXT_SCREEN_SHOT_INCLUDED_
