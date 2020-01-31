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
	
//! TODO: HANDLE UNPACK ALIGNMENT

// change this to produce a CImageData backed by a mapped GPU buffer
core::smart_refctd_ptr<video::IDriverFence> createScreenShot(video::IDriver* driver, core::rect<uint32_t> sourceRect, asset::E_FORMAT _outFormat, video::IGPUBuffer* destination, size_t destOffset=0ull, bool implicitflush=true)
{
	// will change this, https://github.com/buildaworldnet/IrrlichtBAW/issues/148
	if (isBlockCompressionFormat(_outFormat))
		return nullptr;

	GLenum colorformat=GL_INVALID_ENUM, type=GL_INVALID_ENUM;
	video::COpenGLTexture::getOpenGLFormatAndParametersFromColorFormat(_outFormat,colorformat,type);

	video::COpenGLExtensionHandler::extGlBindBuffer(GL_PIXEL_PACK_BUFFER, static_cast<video::COpenGLBuffer*>(destination)->getOpenGLName());
	glReadPixels(sourceRect.UpperLeftCorner.X, sourceRect.UpperLeftCorner.Y, sourceRect.getWidth(), sourceRect.getHeight(), colorformat, type, reinterpret_cast<void*>(destOffset));
	video::COpenGLExtensionHandler::extGlBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

	return driver->placeFence(implicitflush);
}

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
															colorformat,type,destination->getSize()-destOffset,reinterpret_cast<void*>(destOffset));
	video::COpenGLExtensionHandler::extGlBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

	return driver->placeFence(implicitflush);
}

template<typename PathOrFile>
void writeBufferAsImageToFile(asset::IAssetManager* mgr, const PathOrFile& _outFile, core::vector2d<uint32_t> _size, asset::E_FORMAT _format, video::IGPUBuffer* buff, size_t offset=0ull, bool flipY=true)
{
	const uint32_t zero[3] = { 0,0,0 };
	const uint32_t sizeArray[3] = { _size.X,_size.Y,1u };
	auto img = core::make_smart_refctd_ptr<asset::CImageData>(nullptr, zero, sizeArray, 0u, _format);

	//! Wonder if we'll need it after Vulkan ?
	const auto rowSize = (img->getBytesPerPixel()*sizeArray[0]).getRoundedUpInteger();
	const auto imagePitch = img->getPitchIncludingAlignment();
	const uint8_t* inData = reinterpret_cast<const uint8_t*>(buff->getBoundMemory()->getMappedPointer());
	uint8_t* outData = reinterpret_cast<uint8_t*>(img->getData())+imagePitch*(flipY ? (sizeArray[1]-1u):0u);
	for (uint32_t y=0u; y<sizeArray[1]; y++)
	{
		std::move(inData,inData+rowSize,outData);
		inData += imagePitch;
		if (flipY)
			outData -= imagePitch;
		else
			outData += imagePitch;
	}

	asset::IAssetWriter::SAssetWriteParams wparams(img.get());
	mgr->writeAsset(_outFile, wparams);
}


template<typename PathOrFile>
void dirtyCPUStallingScreenshot(video::IVideoDriver* driver, asset::IAssetManager* assetManager, const PathOrFile& _outFile, core::rect<uint32_t> sourceRect, asset::E_FORMAT _format, bool flipY=true)
{
	auto buff = core::smart_refctd_ptr<video::IGPUBuffer>(driver->createDownStreamingGPUBufferOnDedMem((asset::getBytesPerPixel(_format) * sourceRect.getArea()).getIntegerApprox()), core::dont_grab); // TODO
	buff->getBoundMemory()->mapMemoryRange(video::IDriverMemoryAllocation::EMCAF_READ,{0u,buff->getSize()});

	auto fence = ext::ScreenShot::createScreenShot(driver, sourceRect, _format, buff.get());
	while (fence->waitCPU(1000ull, fence->canDeferredFlush()) == video::EDFR_TIMEOUT_EXPIRED) {}
	ext::ScreenShot::writeBufferAsImageToFile(assetManager, _outFile, {sourceRect.getWidth(),sourceRect.getHeight()}, _format, buff.get(), 0ull, flipY);
}

template<typename PathOrFile>
void dirtyCPUStallingScreenshot(video::IVideoDriver* driver, asset::IAssetManager* assetManager, const PathOrFile& _outFile, video::ITexture* source, uint32_t sourceMipLevel = 0u, bool flipY=true)
{
	auto texSize = source->getSize();

	auto buff = core::smart_refctd_ptr<video::IGPUBuffer>(driver->createDownStreamingGPUBufferOnDedMem((source->getPitch()*texSize[1]).getIntegerApprox()), core::dont_grab); // TODO
	buff->getBoundMemory()->mapMemoryRange(video::IDriverMemoryAllocation::EMCAF_READ,{0u,buff->getSize()});

	auto fence = ext::ScreenShot::createScreenShot(driver, source, buff.get(), sourceMipLevel);
	while (fence->waitCPU(1000ull, fence->canDeferredFlush()) == video::EDFR_TIMEOUT_EXPIRED) {}
	ext::ScreenShot::writeBufferAsImageToFile(assetManager, _outFile, { texSize[0],texSize[1] }, source->getColorFormat(), buff.get(), 0ull, flipY);
}


} // namespace ScreenShot
} // namespace ext
} // namespace irr

#endif // _IRR_EXT_SCREEN_SHOT_INCLUDED_
