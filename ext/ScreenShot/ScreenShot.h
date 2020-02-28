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
core::smart_refctd_ptr<video::IDriverFence> createScreenShot(video::IDriver* driver, video::IGPUImage* source, video::IGPUBuffer* destination, uint32_t sourceMipLevel=0u, size_t destOffset=0ull, bool implicitflush=true)
{
	// will change this, https://github.com/buildaworldnet/IrrlichtBAW/issues/148
	if (isBlockCompressionFormat(source->getCreationParameters().format))
		return nullptr;

	auto extent = source->getMipSize(sourceMipLevel);
	video::IGPUImage::SBufferCopy pRegions[1u] = { {destOffset,extent.x,extent.y,{static_cast<asset::IImage::E_ASPECT_FLAGS>(0u),sourceMipLevel,0u,1u},{0u,0u,0u},{extent.x,extent.y,extent.z}} };
	driver->copyImageToBuffer(source,destination,1u,pRegions);

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
void dirtyCPUStallingScreenshot(IrrlichtDevice* device, const PathOrFile& _outFile, video::IGPUImage* source, uint32_t sourceMipLevel = 0u, bool flipY=true)
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
