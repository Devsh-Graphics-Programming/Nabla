// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_IMAGE_MANIPULATOR_H_INCLUDED__
#define __I_IMAGE_MANIPULATOR_H_INCLUDED__

#include "irr/core/core.h"

#include "irr/asset/filters/CCopyImageFilter.h"

namespace irr
{
namespace asset
{

// remember about sampler wrap modes
class CPaddedCopyImageFilter : public CCopyImageFilter
{
	public:
};

// respecifies the image in terms of the least amount of region entries
class CFlattenRegionsImageFilter; // note: make an option that tries to reuse a buffer
using CBufferToImageCopyFilter = CFlattenRegionsImageFilter;

// scaled copies with filters
class CBlitImageFilter;

// specialized case of CBlitImageFilter
class CMipMapGenerationImageFilter;

} // end namespace asset
} // end namespace irr

#endif

/*

uint8_t* convertR8SRGBdataIntoRGB8SRGBAAndGetIt(const void* redChannelDataLayer, const core::smart_refctd_ptr<ICPUImage>& image, const irr::asset::IImage::SBufferCopy& region)	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\asset\IImageLoader.h	36	113
m_physAddrTex = ICPUImage::create(std::move(params));	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\asset\ITexturePacker.h	161	29
m_pageTable = ICPUImage::create(std::move(params));	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\asset\ITexturePacker.h	184	27
page_tab_offset_t pack(const ICPUImage* _img, const ICPUImage::SSubresourceRange& _subres)	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\asset\ITexturePacker.h	257	34
auto image = ICPUImage::create(std::move(imageInfo));	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\CGLILoader.cpp	131	17
auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(imageInfo.mipLevels);	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\CGLILoader.cpp	133	84
bool createAndWriteImage(std::array<ilmType*, availableChannels>& pixelsArrayIlm, const asset::ICPUImage* image, const char* fileName)	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\CImageWriterOpenEXR.cpp	53	98
const asset::ICPUImage* image = IAsset::castDown<ICPUImage>(_params.rootAsset);	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\CImageWriterOpenEXR.cpp	136	17


auto view = core::make_smart_refctd_ptr<ICPUImageView>(std::move(viewParams));	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\ext\MitsubaLoader\CMitsubaLoader.cpp	1286	42
auto pgtabView = ICPUImageView::create(std::move(params));	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\ext\MitsubaLoader\CMitsubaLoader.cpp	1614	20
auto physPgTexView = ICPUImageView::create(std::move(params));	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\ext\MitsubaLoader\CMitsubaLoader.cpp	1634	24

auto dummy2dImgView = core::make_smart_refctd_ptr<asset::ICPUImageView>(std::move(info));	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\IAssetManager.cpp	396	66


ITexturePacker.h	231	                 reinterpret_cast<uint8_t*>(m_pageTable->getBuffer()->getPointer()) + m_pageTable->getRegions().begin()[i].bufferOffset 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\asset
ITexturePacker.h	233	             const uint32_t pgtPitch = m_pageTable->getRegions().begin()[i].bufferRowLength; 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\asset
ITexturePacker.h	283	                 reinterpret_cast<uint8_t*>(m_pageTable->getBuffer()->getPointer()) + m_pageTable->getRegions().begin()[i].bufferOffset 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\asset
ITexturePacker.h	285	             const uint32_t pgtPitch = m_pageTable->getRegions().begin()[i].bufferRowLength; 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\asset
ITexturePacker.h	286	             const uint32_t pgtH = m_pageTable->getRegions().begin()[i].imageExtent.height; 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\asset
ITexturePacker.h	322	                     for (const auto& reg : _img->getRegions()) 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\asset

SRC:

CGLIWriter.cpp	155	    for (auto region = image->getRegions().begin(); region != image->getRegions().end(); ++region) 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset
CGraphicsPipelineLoaderMTL.cpp	964	         const size_t alignment = 1u<<core::findLSB(images[CMTLPipelineMetadata::EMP_REFL_POSX]->getRegions().begin()->bufferRowLength); 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset
CGraphicsPipelineLoaderMTL.cpp	975	                 assert(images[i]->getRegions().size()==1ull); 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset
CGraphicsPipelineLoaderMTL.cpp	977	                 regions_.push_back(images[i]->getRegions().begin()[0]); 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset
CGraphicsPipelineLoaderMTL.cpp	994	                 const void* src = reinterpret_cast<const uint8_t*>(images[i]->getBuffer()->getPointer()) + images[i]->getRegions().begin()[0].bufferOffset; 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset
CImageWriterOpenEXR.cpp	79	    for (auto region = image->getRegions().begin(); region != image->getRegions().end(); ++region) 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset

*/
