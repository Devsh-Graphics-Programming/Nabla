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


// runtime polymorphic
class IImageFilter
{
	public:
		class IState
		{
			public:
				struct Subsection
				{
					VkExtent3D	extent = {0u,0u,0u};
					uint32_t	layerCount = 0u;
					uint32_t	levelCount = 0u;
				};
				struct Offsets
				{
					VkOffset3D	texelOffset = { 0u,0u,0u };
					uint32_t	baseArrayLayer = 0u;
					uint32_t	mipLevel = 0u;
				};
				struct ColorValue
				{
					union
					{
						double				asDouble[4];
						core::vectorSIMDf	asFloat;
						core::vectorSIMD32u asUint;
						core::vectorSIMD32i asInt;
						uint16_t			asUShort[4];
						int16_t				asShort[4];
						uint8_t				asUByte[4];
						int8_t				asByte[4];
					};
				};
				
			protected:
				virtual ~IState() = 0;
		};		

        //
		virtual bool pValidate(IState* state) const = 0;
		
		//
		virtual bool pExecute(IState* state) const = 0;
		
	protected:
	    virtual ~IImageFilter() = 0;
};

IImageFilter::IState::~IState() {}
IImageFilter::~IImageFilter() {}

// static polymorphic
template<typename CRTP>
class CImageFilter : public IImageFilter
{
	public:
		static inline bool validate(IState* state)
		{
			return CRTP::validate(static_cast<typename CRTP::state_type*>(state));
		}
		
		inline bool pValidate(IState* state) const override
		{
			return validate(state);
		}

		static inline bool execute(IState* state)
		{
			return CRTP::execute(static_cast<typename CRTP::state_type*>(state));
		}

		inline bool pExecute(IState* state) const override
		{
			return execute(state);
		}
};

class CBasicInImageFilterCommon
{
	public:
		class CState : public IImageFilter::IState
		{
			public:
				virtual ~CState() {}

				Subsection			subsection = {};
				const ICPUImage*	inImage = nullptr;
				Offsets				inOffsets = {};
		};
		using state_type = CState;

		static inline bool validate(CState* state)
		{
			if (!state)
				return nullptr;

			const auto& inCreationParams = state->inImage->getCreationParameters();
			// TODO: Extra epic validation

			return true;
		}

	protected:
		virtual ~CBasicInImageFilterCommon() = 0;
};
class CBasicOutImageFilterCommon
{
	public:
		class CState : public IImageFilter::IState
		{
			public:
				virtual ~CState() {}

				Subsection	subsection = {};
				ICPUImage*	outImage = nullptr;
				Offsets		outOffsets = {};
		};
		using state_type = CState;

		static inline bool validate(CState* state)
		{
			if (!state)
				return nullptr;

			const auto& outCreationParams = state->outImage->getCreationParameters();
			// TODO: Extra epic validation

			return true;
		}

	protected:
		virtual ~CBasicOutImageFilterCommon() = 0;
};
class CBasicInOutImageFilterCommon
{
	public:
		class CState : public IImageFilter::IState
		{
			public:
				virtual ~CState() {}

				Subsection			subsection = {};
				const ICPUImage*	inImage = nullptr;
				ICPUImage*			outImage = nullptr;
				Offsets				inOffsets = {};
				Offsets				outOffsets = {};
		};
		using state_type = CState;

		static inline bool validate(CState* state)
		{
			if (!state)
				return nullptr;

			const auto& inCreationParams = state->inImage->getCreationParameters();
			const auto& outCreationParams = state->outImage->getCreationParameters();
			// TODO: Extra epic validation

			return true;
		}
		

	protected:
		virtual ~CBasicInOutImageFilterCommon() = 0;
};
// will probably need some per-pixel helper class/functions (that can run a templated functor per-pixel to reduce code clutter)

// fill a section of the image with a uniform value
class CFillImageFilter : public CImageFilter<CFillImageFilter>
{
	public:
		virtual ~CFillImageFilter() {}

		class CState : public CBasicOutImageFilterCommon::state_type
		{
			public:
				ColorValue fillValue;

				virtual ~CState() {}
		};
		using state_type = CState;

		static inline bool validate(CState* state)
		{
			return CBasicOutImageFilterCommon::validate(state);
		}

		static inline bool execute(CState* state)
		{
			if (!validate(state))
				return false;

			// do the per-pixel filling

			return true;
		}
};

// convert between image formats
class CConvertFormatImageFilter : public CImageFilter<CConvertFormatImageFilter>
{
	public:
		virtual ~CConvertFormatImageFilter() {}

		class CState : public CBasicInOutImageFilterCommon::state_type
		{
			public:
				virtual ~CState() {}
		};
		using state_type = CState;

		static inline bool validate(CState* state)
		{
			return CBasicInOutImageFilterCommon::validate(state);
		}

		static inline bool execute(CState* state)
		{
			if (!validate(state))
				return false;

			// do the per-pixel convert

			return true;
		}
};

class CCopyImageFilter : public CImageFilter<CCopyImageFilter>
{
	public:
		virtual ~CCopyImageFilter() {}

		class CState : public CBasicInOutImageFilterCommon::state_type
		{
			public:
				virtual ~CState() {}
		};
		using state_type = CState;

		static inline bool validate(CState* state)
		{
			if (!CBasicInOutImageFilterCommon::validate(state))
				return false;

			const auto& inCreationParams = state->inImage->getCreationParameters();
			const auto& outCreationParams = state->outImage->getCreationParameters();

			if (getTexelBlockSize(inCreationParams.format)!=getTexelBlockSize(outCreationParams))
				return false;

			if (getTexelBlockBytesize(inCreationParams.format)!=getTexelBlockBytesize(outCreationParams))
				return false;

			return true;
		}

		static inline bool execute(CState* state)
		{
			if (!validate(state))
				return false;

			// do the per-pixel copy

			return true;
		}
};


// respecifies the image in terms of the least amount of region entries
class CFlattenRegionsImageFilter; // note: make an option that tries to reuse a buffer
using CBufferToImageCopyFilter = CFlattenRegionsImageFilter;

// lets you turn a complex image to a buffer
class CImageToBufferCopyFilter;

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
bool createAndWriteImage(std::array<ilmType*, availableChannels>& pixelsArrayIlm, const asset::ICPUImage* image, const char* fileName)	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\CImageWriterOpenEXR.cpp	53	98
const asset::ICPUImage* image = IAsset::castDown<ICPUImage>(_params.rootAsset);	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\CImageWriterOpenEXR.cpp	136	17


auto view = core::make_smart_refctd_ptr<ICPUImageView>(std::move(viewParams));	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\ext\MitsubaLoader\CMitsubaLoader.cpp	1286	42
auto pgtabView = ICPUImageView::create(std::move(params));	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\ext\MitsubaLoader\CMitsubaLoader.cpp	1614	20
auto physPgTexView = ICPUImageView::create(std::move(params));	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\ext\MitsubaLoader\CMitsubaLoader.cpp	1634	24

auto gpuImgViews = getGPUObjectsFromAssets<asset::ICPUImageView>(cpuImgViews.data(), cpuImgViews.data()+cpuImgViews.size(), _params);	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\video\IGPUObjectFromAssetConverter.h	748	55
auto imageView = ICPUImageView::create(std::move(imageViewInfo));	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\CGLILoader.cpp	213	21
const asset::ICPUImageView* imageView = IAsset::castDown<ICPUImageView>(_params.rootAsset);	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\CGLIWriter.cpp	48	17
bool CGLIWriter::writeGLIFile(io::IWriteFile* file, const asset::ICPUImageView* imageView)	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\CGLIWriter.cpp	61	68
bool writeGLIFile(io::IWriteFile* file, const asset::ICPUImageView* imageView);	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\CGLIWriter.h	46	56
if (auto view = getDefaultAsset<ICPUImageView,IAsset::ET_IMAGE_VIEW>(viewCacheKey.c_str(), m_assetMgr))	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\CGraphicsPipelineLoaderMTL.cpp	1027	41
views[i] = ICPUImageView::create(std::move(viewParams));	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\CGraphicsPipelineLoaderMTL.cpp	1048	20
using image_views_set_t = std::array<core::smart_refctd_ptr<ICPUImageView>, CMTLPipelineMetadata::EMP_REFL_POSX + 1u>;	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\CGraphicsPipelineLoaderMTL.h	58	69
auto dummy2dImgView = core::make_smart_refctd_ptr<asset::ICPUImageView>(std::move(info));	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset\IAssetManager.cpp	396	66


ITexturePacker.h	231	                 reinterpret_cast<uint8_t*>(m_pageTable->getBuffer()->getPointer()) + m_pageTable->getRegions().begin()[i].bufferOffset 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\asset
ITexturePacker.h	233	             const uint32_t pgtPitch = m_pageTable->getRegions().begin()[i].bufferRowLength; 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\asset
ITexturePacker.h	283	                 reinterpret_cast<uint8_t*>(m_pageTable->getBuffer()->getPointer()) + m_pageTable->getRegions().begin()[i].bufferOffset 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\asset
ITexturePacker.h	285	             const uint32_t pgtPitch = m_pageTable->getRegions().begin()[i].bufferRowLength; 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\asset
ITexturePacker.h	286	             const uint32_t pgtH = m_pageTable->getRegions().begin()[i].imageExtent.height; 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\asset
ITexturePacker.h	322	                     for (const auto& reg : _img->getRegions()) 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\asset
IGPUObjectFromAssetConverter.h	347	   auto regions = cpuimg->getRegions(); 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\video
IGPUObjectFromAssetConverter.h	352	    m_driver->copyBufferToImage(tmpBuff.get(),gpuimg.get(),count,cpuimg->getRegions().begin()); 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\include\irr\video

SRC:

CGLIWriter.cpp	155	    for (auto region = image->getRegions().begin(); region != image->getRegions().end(); ++region) 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset
CGraphicsPipelineLoaderMTL.cpp	964	         const size_t alignment = 1u<<core::findLSB(images[CMTLPipelineMetadata::EMP_REFL_POSX]->getRegions().begin()->bufferRowLength); 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset
CGraphicsPipelineLoaderMTL.cpp	975	                 assert(images[i]->getRegions().size()==1ull); 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset
CGraphicsPipelineLoaderMTL.cpp	977	                 regions_.push_back(images[i]->getRegions().begin()[0]); 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset
CGraphicsPipelineLoaderMTL.cpp	994	                 const void* src = reinterpret_cast<const uint8_t*>(images[i]->getBuffer()->getPointer()) + images[i]->getRegions().begin()[0].bufferOffset; 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset
CImageWriterOpenEXR.cpp	79	    for (auto region = image->getRegions().begin(); region != image->getRegions().end(); ++region) 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset

*/
