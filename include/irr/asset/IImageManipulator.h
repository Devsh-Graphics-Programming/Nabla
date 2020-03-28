// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_IMAGE_MANIPULATOR_H_INCLUDED__
#define __I_IMAGE_MANIPULATOR_H_INCLUDED__

#include "irr/core/core.h"

#include "irr/asset/ICPUImage.h"

namespace irr
{
namespace asset
{

class IImageManipulator : public IReferenceCounted
{
	public:
};

} // end namespace asset
} // end namespace irr

#endif

/*

uint8_t* convertR8SRGBdataIntoRGB8SRGBAAndGetIt(const void* redChannelDataLayer, const core::smart_refctd_ptr<ICPUImage>& image, const irr::asset::IImage::SBufferCopy& region)	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\asset\IImageLoader.h	36	113
m_physAddrTex = ICPUImage::create(std::move(params));	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\asset\ITexturePacker.h	161	29
m_pageTable = ICPUImage::create(std::move(params));	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\asset\ITexturePacker.h	184	27
page_tab_offset_t pack(const ICPUImage* _img, const ICPUImage::SSubresourceRange& _subres)	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\asset\ITexturePacker.h	257	34
core::smart_refctd_ptr<ICPUImage> m_physAddrTex;	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\asset\ITexturePacker.h	378	28
core::smart_refctd_ptr<ICPUImage> m_pageTable;	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\asset\ITexturePacker.h	379	28
inline virtual created_gpu_object_array<asset::ICPUImage>				            create(asset::ICPUImage** const _begin, asset::ICPUImage** const _end, const SParams& _params);	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\video\IGPUObjectFromAssetConverter.h	59	50
auto IGPUObjectFromAssetConverter::create(asset::ICPUImage** const _begin, asset::ICPUImage** const _end, const SParams& _params) -> created_gpu_object_array<asset::ICPUImage>	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\video\IGPUObjectFromAssetConverter.h	335	50
auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUImage> >(assetCount);	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\video\IGPUObjectFromAssetConverter.h	338	80
const asset::ICPUImage* cpuimg = _begin[i];	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\video\IGPUObjectFromAssetConverter.h	342	22
core::vector<asset::ICPUImage*> cpuDeps;	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\video\IGPUObjectFromAssetConverter.h	608	25
auto gpuDeps = getGPUObjectsFromAssets<asset::ICPUImage>(cpuDeps.data(), cpuDeps.data() + cpuDeps.size(), _params);	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\video\IGPUObjectFromAssetConverter.h	619	51
auto image = ICPUImage::create(std::move(imageInfo));	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\CGLILoader.cpp	131	17
auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(imageInfo.mipLevels);	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\CGLILoader.cpp	133	84
images[i] = core::smart_refctd_ptr_static_cast<ICPUImage>(bundle.getContents().first[0]);	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\CGraphicsPipelineLoaderMTL.cpp	936	64
auto cubemap = ICPUImage::create(std::move(cubemapParams));	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\CGraphicsPipelineLoaderMTL.cpp	1009	24
using images_set_t = std::array<core::smart_refctd_ptr<ICPUImage>, CMTLPipelineMetadata::EMP_COUNT>;	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\CGraphicsPipelineLoaderMTL.h	57	64
core::vector<core::smart_refctd_ptr<ICPUImage>> images;	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\CImageLoaderOpenEXR.cpp	214	40
auto image = ICPUImage::create(std::move(params));	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\CImageLoaderOpenEXR.cpp	253	19
core::smart_refctd_ptr<ICPUImage> image = nullptr;	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\CImageLoaderPNG.cpp	213	28
image = ICPUImage::create(std::move(imgInfo));	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\CImageLoaderPNG.cpp	237	13
core::smart_refctd_ptr<ICPUImage> image = ICPUImage::create(std::move(imgInfo));	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\CImageLoaderTGA.cpp	199	25
bool createAndWriteImage(std::array<ilmType*, availableChannels>& pixelsArrayIlm, const asset::ICPUImage* image, const char* fileName)	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\CImageWriterOpenEXR.cpp	53	98
const asset::ICPUImage* image = IAsset::castDown<ICPUImage>(_params.rootAsset);	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\CImageWriterOpenEXR.cpp	136	17
bool CImageWriterOpenEXR::writeImageBinary(io::IWriteFile* file, const asset::ICPUImage* image)	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\CImageWriterOpenEXR.cpp	149	81
bool writeImageBinary(io::IWriteFile* file, const asset::ICPUImage* image);	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\CImageWriterOpenEXR.h	44	60
core::smart_refctd_ptr<asset::ICPUImage> dummy2dImage;	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\IAssetManager.cpp	351	35
dummy2dImage = asset::ICPUImage::create(std::move(info));	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\IAssetManager.cpp	370	31


*/
