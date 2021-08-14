/*
MIT License
Copyright (c) 2019 AnastaZIuk
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "CGLIWriter.h"

#ifdef _NBL_COMPILE_WITH_GLI_WRITER_

#include "nbl/asset/filters/CBasicImageFilterCommon.h"
#include "nbl/asset/filters/CSwizzleAndConvertImageFilter.h"

#ifdef _NBL_COMPILE_WITH_GLI_
#include "gli/gli.hpp"
#else
#error "It requires GLI library"
#endif

namespace nbl
{
namespace asset
{
static inline std::pair<gli::texture::format_type, std::array<gli::gl::swizzle, 4>> getTranslatedIRRFormat(const IImageView<ICPUImage>::SCreationParams& params, const system::logger_opt_ptr logger);

static inline bool performSavingAsIFile(gli::texture& texture, system::IFile* file, system::ISystem* sys, const system::logger_opt_ptr logger);

bool CGLIWriter::writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
{
	if (!_override)
		getDefaultOverride(_override);

	SAssetWriteContext ctx{ _params, _file };

	auto imageView = IAsset::castDown<const ICPUImageView>(_params.rootAsset);

	if (!imageView)
		return false;

	system::IFile* file = _override->getOutputFile(_file, ctx, { imageView, 0u });

	if (!file)
		return false;

	return writeGLIFile(file, imageView, _params.logger);
}

bool CGLIWriter::writeGLIFile(system::IFile* file, const asset::ICPUImageView* imageView, const system::logger_opt_ptr logger)
{
	logger.log("WRITING GLI: writing the file %s", system::ILogger::ELL_INFO, file->getFileName().string().c_str());

	const auto& imageViewInfo = imageView->getCreationParameters();
	const auto& imageInfo = imageViewInfo.image->getCreationParameters();
	const auto& image = imageViewInfo.image;

	if (image->getRegions().size() == 0)
	{
		logger.log("WRITING GLI: there is a lack of regions! %s", system::ILogger::ELL_INFO, file->getFileName().string().c_str());
		return false;
	}

	asset::DefaultSwizzle swizzleMapping;
	{
		const auto& swizzle = imageView->getComponents();
		swizzleMapping.swizzle = swizzle;
	}

	const bool isItACubemap = doesItHaveFaces(imageViewInfo.viewType);
	const bool layersFlag = doesItHaveLayers(imageViewInfo.viewType);
		
	const auto texelBlockByteSize = asset::getTexelOrBlockBytesize(imageInfo.format);
	const auto data = reinterpret_cast<const uint8_t*>(image->getBuffer()->getPointer());

	auto getTarget = [&]()
	{
		switch (imageViewInfo.viewType)
		{
			case ICPUImageView::ET_1D: return gli::TARGET_1D;
			case ICPUImageView::ET_1D_ARRAY: return gli::TARGET_1D_ARRAY;
			case ICPUImageView::ET_2D: return gli::TARGET_2D;
			case ICPUImageView::ET_2D_ARRAY: return gli::TARGET_2D_ARRAY;
			case ICPUImageView::ET_3D: return gli::TARGET_3D;
			case ICPUImageView::ET_CUBE_MAP: return gli::TARGET_CUBE;
			case ICPUImageView::ET_CUBE_MAP_ARRAY: return gli::TARGET_CUBE_ARRAY;
			default: assert(0);
		}
		return static_cast<gli::target>(-1); // make Warnings shut up
	};

	auto getFacesAndLayersAmount = [&]()
	{
		size_t layers, faces;
		const auto arrayLayers = imageInfo.arrayLayers;

		if (isItACubemap)
		{
			faces = ((arrayLayers - 1) % 6) + 1;

			if (layersFlag)
				layers = ((arrayLayers - 1) / 6) + 1;
			else
				layers = 1;
		}
		else
		{
			faces = 1;
			layers = arrayLayers;
		}
			
		return std::make_pair(layers, faces);
	};

	auto getCurrentGliLayerAndFace = [&](uint16_t layer)
	{
		static uint16_t gliLayer, gliFace;

		if (isItACubemap)
		{
			gliFace = layer % 6;

			if (layersFlag)
				gliLayer = layer / 6;
			else
				gliLayer = 0;
		}
		else
		{
			gliFace = 0;
			gliLayer = layer;
		}

		return std::make_pair(gliLayer, gliFace);
	};

	auto gliFormatAndSwizzles = getTranslatedIRRFormat(imageViewInfo, logger);
	gli::target gliTarget = getTarget();
	gli::extent3d gliExtent3d = {imageInfo.extent.width, imageInfo.extent.height, imageInfo.extent.depth};
	size_t gliLevels = imageInfo.mipLevels;
	std::pair<size_t, size_t> layersAndFacesAmount = getFacesAndLayersAmount();

	gli::texture texture(gliTarget, gliFormatAndSwizzles.first, gliExtent3d, layersAndFacesAmount.first, layersAndFacesAmount.second, gliLevels, gli::texture::swizzles_type{ gliFormatAndSwizzles.second[0], gliFormatAndSwizzles.second[1], gliFormatAndSwizzles.second[2], gliFormatAndSwizzles.second[3] });
	
	struct State
	{
		uint32_t currentMipLevel;
		core::vectorSIMDu32 outStrides;
		core::vectorSIMDu32 outBlocks;
	} state;
	const bool isBC = asset::isBlockCompressionFormat(imageInfo.format);
	const bool isInteger = asset::isIntegerFormat(imageInfo.format);
	auto writeTexel = [&data,&texelBlockByteSize,getCurrentGliLayerAndFace,&state,&texture,&imageInfo,&swizzleMapping,&isBC,&isInteger](uint32_t ptrOffset, const core::vectorSIMDu32& texelCoord) -> void
	{
		const uint8_t* inData = data+ptrOffset;

		const auto layersData = getCurrentGliLayerAndFace(texelCoord.w);
		const auto gliLayer = layersData.first;
		const auto gliFace = layersData.second;
		uint8_t* outData = reinterpret_cast<uint8_t*>(texture.data(gliLayer,gliFace,state.currentMipLevel));
		outData += asset::IImage::SBufferCopy::getLocalByteOffset(texelCoord, state.outStrides);

		if (isBC)
			memcpy(outData, inData, texelBlockByteSize);
		else
		{
			const void* sourcePixels[] = { inData, nullptr, nullptr, nullptr };
			constexpr uint8_t maxChannels = 4;
			double decodeBuffer[maxChannels] = {};
			double swizzleDecodeBuffer[maxChannels] = {};

			asset::decodePixelsRuntime(imageInfo.format, sourcePixels, decodeBuffer, 0, 0);
			if(isInteger)
				swizzleMapping(reinterpret_cast<uint64_t*>(decodeBuffer), reinterpret_cast<uint64_t*>(swizzleDecodeBuffer));
			else
				swizzleMapping(decodeBuffer, swizzleDecodeBuffer);
			asset::encodePixelsRuntime(imageInfo.format, outData, swizzleDecodeBuffer);
		}
	};
	const TexelBlockInfo blockInfo(imageInfo.format);
	auto updateState = [&state,&blockInfo,texelBlockByteSize,image](const IImage::SBufferCopy& newRegion, const IImage::SBufferCopy* referenceRegion) -> bool
	{
		state.currentMipLevel = referenceRegion->imageSubresource.mipLevel;

		state.outBlocks = blockInfo.convertTexelsToBlocks(image->getMipSize(state.currentMipLevel));
		state.outStrides[0] = texelBlockByteSize;
		state.outStrides[1] = state.outBlocks[0]*texelBlockByteSize;
		state.outStrides[2] = state.outBlocks[1]*state.outStrides[1];
		state.outStrides[3] = 0; // GLI function gets the correct layer by itself
		return true;
	};

	const auto& regions = image->getRegions();
	CBasicImageFilterCommon::executePerRegion<decltype(writeTexel),decltype(updateState)>(image.get(),writeTexel,regions.begin(),regions.end(),updateState);

	return performSavingAsIFile(texture, file, m_system.get(), logger);
}
bool performSavingAsIFile(gli::texture& texture, system::IFile* file, system::ISystem* sys, const system::logger_opt_ptr logger)
{
	if (texture.empty())
		return false;

	const auto fileName = std::string(file->getFileName().string());
	std::vector<char> memory;
	bool properlyStatus = false;

	if (fileName.rfind(".dds") != std::string::npos)
		properlyStatus = save_dds(texture, memory);
	if (!properlyStatus && fileName.rfind(".kmg") != std::string::npos)
		properlyStatus = save_kmg(texture, memory);
	if (!properlyStatus && fileName.rfind(".ktx") != std::string::npos)
		properlyStatus = save_ktx(texture, memory);

	if (properlyStatus)
	{
		system::future<size_t> future;
		file->write(future, memory.data(), 0, memory.size());
		future.get();
		return true;
	}
	else
	{
		logger.log("WRITING GLI: failed to save the file %s", system::ILogger::ELL_ERROR, file->getFileName().string().c_str());
		return false;
	}
}

inline std::pair<gli::texture::format_type, std::array<gli::gl::swizzle, 4>> getTranslatedIRRFormat(const IImageView<ICPUImage>::SCreationParams& params, const system::logger_opt_ptr logger)
{
	using namespace gli;
	std::array<gli::gl::swizzle, 4> compomentMapping;

	static const core::unordered_map<ICPUImageView::SComponentMapping::E_SWIZZLE, gli::gl::swizzle> swizzlesMappingAPI(
	{
		std::make_pair(ICPUImageView::SComponentMapping::ES_R, gl::SWIZZLE_RED),
		std::make_pair(ICPUImageView::SComponentMapping::ES_G, gl::SWIZZLE_GREEN),
		std::make_pair(ICPUImageView::SComponentMapping::ES_B, gl::SWIZZLE_BLUE),
		std::make_pair(ICPUImageView::SComponentMapping::ES_A, gl::SWIZZLE_ALPHA),
		std::make_pair(ICPUImageView::SComponentMapping::ES_ONE, gl::SWIZZLE_ONE),
		std::make_pair(ICPUImageView::SComponentMapping::ES_ZERO, gl::SWIZZLE_ZERO)
	});

	auto getMappedSwizzle = [&](const ICPUImageView::SComponentMapping::E_SWIZZLE& currentSwizzleToCheck) -> gli::gl::swizzle
	{
		for (auto& mappedSwizzle : swizzlesMappingAPI)
			if (currentSwizzleToCheck == mappedSwizzle.first)
				return mappedSwizzle.second;
		assert(false);
		return gli::gl::swizzle::SWIZZLE_ZERO;
	};

	compomentMapping[0] = getMappedSwizzle(params.components.r);
	compomentMapping[1] = getMappedSwizzle(params.components.g);
	compomentMapping[2] = getMappedSwizzle(params.components.b);
	compomentMapping[3] = getMappedSwizzle(params.components.a);

	auto getTranslatedFinalFormat = [&](const gli::texture::format_type& format = FORMAT_UNDEFINED, const std::string_view& specialErrorOnUnknown = "Unsupported format!")
	{
		if (format == FORMAT_UNDEFINED)
		{
			logger.log(("WRITING GLI: " + std::string(specialErrorOnUnknown)).c_str(), system::ILogger::ELL_ERROR);
		}

		return std::make_pair(format, compomentMapping);
	};

	switch (params.format)
	{
		// "///" means a format doesn't fit in Power-Of-Two for a texel/block (there is an exception in Power-Of-Two rule - 24bit RGB basic format)
			
		case EF_R8G8B8_UNORM: return getTranslatedFinalFormat(FORMAT_RGB8_UNORM_PACK8);			//GL_RGB
		case EF_B8G8R8_UNORM: return getTranslatedFinalFormat(FORMAT_BGR8_UNORM_PACK8);			//GL_BGR
		case EF_R8G8B8A8_UNORM: return getTranslatedFinalFormat(FORMAT_RGBA8_UNORM_PACK8);		//GL_RGBA
		case EF_B8G8R8A8_UNORM: return getTranslatedFinalFormat(FORMAT_BGRA8_UNORM_PACK8);		//GL_BGRA
				
		// unorm formats
		case EF_R8_UNORM: return getTranslatedFinalFormat(FORMAT_R8_UNORM_PACK8);			//GL_R8
		case EF_R8G8_UNORM: return getTranslatedFinalFormat(FORMAT_RG8_UNORM_PACK8);		//GL_RG8
		//case EF_R8G8B8_UNORM: return getTranslatedFinalFormat();		//GL_RGB8
		//case EF_R8G8B8A8_UNORM: return getTranslatedFinalFormat();		//GL_RGBA8

		case EF_R16_UNORM: return getTranslatedFinalFormat(FORMAT_R16_UNORM_PACK16);		//GL_R16
		case EF_R16G16_UNORM: return getTranslatedFinalFormat(FORMAT_RG16_UNORM_PACK16);		//GL_RG16
		///case EF_R16G16B16_UNORM: return getTranslatedFinalFormat(FORMAT_RGB16_UNORM_PACK16);		//GL_RGB16
		case EF_R16G16B16A16_UNORM: return getTranslatedFinalFormat(FORMAT_RGBA16_UNORM_PACK16);		//GL_RGBA16

		case EF_A2R10G10B10_UNORM_PACK32: return getTranslatedFinalFormat(FORMAT_RGB10A2_UNORM_PACK32);	//GL_RGB10_A2

		// snorm formats
		case EF_R8_SNORM: return getTranslatedFinalFormat(FORMAT_R8_SNORM_PACK8);			//GL_R8_SNORM
		case EF_R8G8_SNORM: return getTranslatedFinalFormat(FORMAT_RG8_SNORM_PACK8);		//GL_RG8_SNORM
		case EF_R8G8B8_SNORM: return getTranslatedFinalFormat(FORMAT_RGB8_SNORM_PACK8);		//GL_RGB8_SNORM
		case EF_R8G8B8A8_SNORM: return getTranslatedFinalFormat(FORMAT_BGR8_SNORM_PACK8);		//GL_RGBA8_SNORM

		case EF_R16_SNORM: return getTranslatedFinalFormat(FORMAT_R16_SNORM_PACK16);		//GL_R16_SNORM
		case EF_R16G16_SNORM: return getTranslatedFinalFormat(FORMAT_RG16_SNORM_PACK16);		//GL_RG16_SNORM
		///case EF_R16G16B16_SNORM: return getTranslatedFinalFormat(FORMAT_RGB16_SNORM_PACK16);		//GL_RGB16_SNORM
		case EF_R16G16B16A16_SNORM: return getTranslatedFinalFormat(FORMAT_RGBA16_SNORM_PACK16);		//GL_RGBA16_SNORM

		// unsigned integer formats
		case EF_R8_UINT: return getTranslatedFinalFormat(FORMAT_R8_UINT_PACK8);				//GL_R8UI
		case EF_R8G8_UINT: return getTranslatedFinalFormat(FORMAT_RG8_UINT_PACK8);				//GL_RG8UI
		case EF_R8G8B8_UINT: return getTranslatedFinalFormat(FORMAT_RGB8_UINT_PACK8);			//GL_RGB8UI
		case EF_R8G8B8A8_UINT: return getTranslatedFinalFormat(FORMAT_RGBA8_UINT_PACK8);			//GL_RGBA8UI

		case EF_R16_UINT: return getTranslatedFinalFormat(FORMAT_R16_UINT_PACK16);				//GL_R16UI
		case EF_R16G16_UINT: return getTranslatedFinalFormat(FORMAT_RG16_UINT_PACK16);			//GL_RG16UI
		///case EF_R16G16B16_UINT: return getTranslatedFinalFormat(FORMAT_RGB16_UINT_PACK16);			//GL_RGB16UI
		case EF_R16G16B16A16_UINT: return getTranslatedFinalFormat(FORMAT_RGBA16_UINT_PACK16);			//GL_RGBA16UI

		case EF_R32_UINT: return getTranslatedFinalFormat(FORMAT_R32_UINT_PACK32);				//GL_R32UI
		case EF_R32G32_UINT: return getTranslatedFinalFormat(FORMAT_RG32_UINT_PACK32);			//GL_RG32UI
		///case EF_R32G32B32_UINT: return getTranslatedFinalFormat(FORMAT_RGB32_UINT_PACK32);			//GL_RGB32UI
		case EF_R32G32B32A32_UINT: return getTranslatedFinalFormat(FORMAT_RGBA32_UINT_PACK32);			//GL_RGBA32UI

		case EF_A2R10G10B10_UINT_PACK32: return getTranslatedFinalFormat(FORMAT_RGB10A2_UINT_PACK32);			//GL_RGB10_A2UI
		case EF_A2R10G10B10_SINT_PACK32: return getTranslatedFinalFormat(FORMAT_RGB10A2_SINT_PACK32);	//GL_RGB10_A2I

		// signed integer formats
		case EF_R8_SINT: return getTranslatedFinalFormat(FORMAT_R8_SINT_PACK8);				//GL_R8I
		case EF_R8G8_SINT: return getTranslatedFinalFormat(FORMAT_RG8_SINT_PACK8);				//GL_RG8I
		case EF_R8G8B8_SINT: return getTranslatedFinalFormat(FORMAT_RGB8_SINT_PACK8);			//GL_RGB8I
		case EF_R8G8B8A8_SINT: return getTranslatedFinalFormat(FORMAT_RGBA8_SINT_PACK8);			//GL_RGBA8I

		case EF_R16_SINT: return getTranslatedFinalFormat(FORMAT_R16_SINT_PACK16);				//GL_R16I
		case EF_R16G16_SINT: return getTranslatedFinalFormat(FORMAT_RG16_SINT_PACK16);			//GL_RG16I
		///case EF_R16G16B16_SINT: return getTranslatedFinalFormat(FORMAT_RGB16_SINT_PACK16);			//GL_RGB16I
		case EF_R16G16B16A16_SINT: return getTranslatedFinalFormat(FORMAT_RGBA16_SINT_PACK16);			//GL_RGBA16I

		case EF_R32_SINT: return getTranslatedFinalFormat(FORMAT_R32_SINT_PACK32);				//GL_R32I
		case EF_R32G32_SINT: return getTranslatedFinalFormat(FORMAT_RG32_SINT_PACK32);			//GL_RG32I
		///case EF_R32G32B32_SINT: return getTranslatedFinalFormat(FORMAT_RGB32_SINT_PACK32);			//GL_RGB32I
		case EF_R32G32B32A32_SINT: return getTranslatedFinalFormat(FORMAT_RGBA32_SINT_PACK32);			//GL_RGBA32I

		// floating formats
		case EF_R16_SFLOAT: return getTranslatedFinalFormat(FORMAT_R16_SFLOAT_PACK16);				//GL_R16F
		case EF_R16G16_SFLOAT: return getTranslatedFinalFormat(FORMAT_RG16_SFLOAT_PACK16);			//GL_RG16F
		///case EF_R16G16B16_SFLOAT: return getTranslatedFinalFormat(FORMAT_RGB16_SFLOAT_PACK16);			//GL_RGB16F
		case EF_R16G16B16A16_SFLOAT: return getTranslatedFinalFormat(FORMAT_RGBA16_SFLOAT_PACK16);			//GL_RGBA16F

		case EF_R32_SFLOAT: return getTranslatedFinalFormat(FORMAT_R32_SFLOAT_PACK32);				//GL_R32F
		case EF_R32G32_SFLOAT: return getTranslatedFinalFormat(FORMAT_RG32_SFLOAT_PACK32);			//GL_RG32F
		///case EF_R32G32B32_SFLOAT: return getTranslatedFinalFormat(FORMAT_RGB32_SFLOAT_PACK32);			//GL_RGB32F
		case EF_R32G32B32A32_SFLOAT: return getTranslatedFinalFormat(FORMAT_RGBA32_SFLOAT_PACK32);			//GL_RGBA32F

		case EF_R64_SFLOAT: return getTranslatedFinalFormat(FORMAT_R64_SFLOAT_PACK64);			//GL_R64F
		case EF_R64G64_SFLOAT: return getTranslatedFinalFormat(FORMAT_RG64_SFLOAT_PACK64);		//GL_RG64F
		///case EF_R64G64B64_SFLOAT: return getTranslatedFinalFormat(FORMAT_RGB64_SFLOAT_PACK64);		//GL_RGB64F
		case EF_R64G64B64A64_SFLOAT: return getTranslatedFinalFormat(FORMAT_RGBA64_SFLOAT_PACK64);		//GL_RGBA64F

		// sRGB formats
		case EF_R8_SRGB: return getTranslatedFinalFormat(FORMAT_R8_SRGB_PACK8);				//GL_SR8_EXT
		case EF_R8G8_SRGB: return getTranslatedFinalFormat(FORMAT_RG8_SRGB_PACK8);				//GL_SRG8_EXT
		case EF_R8G8B8_SRGB: return getTranslatedFinalFormat(FORMAT_RGB8_SRGB_PACK8);			//GL_SRGB8
		case EF_R8G8B8A8_SRGB: return getTranslatedFinalFormat(FORMAT_RGBA8_SRGB_PACK8);		//GL_SRGB8_ALPHA8
			
		case EF_BC1_RGB_SRGB_BLOCK: return getTranslatedFinalFormat(FORMAT_RGB_DXT1_SRGB_BLOCK8);					//GL_COMPRESSED_SRGB_S3TC_DXT1_EXT
		case EF_BC1_RGBA_SRGB_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_DXT1_SRGB_BLOCK8);				//GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT
		case EF_BC2_SRGB_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_DXT3_SRGB_BLOCK16);				//GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT
		case EF_BC3_SRGB_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_DXT5_SRGB_BLOCK16);				//GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT
		case EF_BC7_SRGB_BLOCK: return getTranslatedFinalFormat(FORMAT_RGB_BP_UFLOAT_BLOCK16 /*there should be BP_UNORM, but no provided for RGB*/);	//GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM
		case EF_ETC2_R8G8B8_SRGB_BLOCK: return getTranslatedFinalFormat(FORMAT_RGB_ETC2_SRGB_BLOCK8);						//GL_COMPRESSED_SRGB8_ETC2
		case EF_ETC2_R8G8B8A1_SRGB_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ETC2_SRGB_BLOCK8);	//GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2
		case EF_ETC2_R8G8B8A8_SRGB_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ETC2_SRGB_BLOCK8);			//GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC
		case EF_ASTC_4x4_SRGB_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ASTC_4X4_SRGB_BLOCK16);		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR
		case EF_ASTC_5x4_SRGB_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ASTC_5X4_SRGB_BLOCK16);		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR
		case EF_ASTC_5x5_SRGB_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ASTC_5X5_SRGB_BLOCK16);		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR
		case EF_ASTC_6x5_SRGB_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ASTC_6X5_SRGB_BLOCK16);		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR
		case EF_ASTC_6x6_SRGB_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ASTC_6X6_SRGB_BLOCK16);		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR
		case EF_ASTC_8x5_SRGB_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ASTC_8X5_SRGB_BLOCK16);		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR
		case EF_ASTC_8x6_SRGB_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ASTC_8X6_SRGB_BLOCK16);		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR
		case EF_ASTC_8x8_SRGB_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ASTC_8X8_SRGB_BLOCK16);		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR
		case EF_ASTC_10x5_SRGB_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ASTC_10X5_SRGB_BLOCK16);		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR
		case EF_ASTC_10x6_SRGB_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ASTC_10X6_SRGB_BLOCK16);		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR
		case EF_ASTC_10x8_SRGB_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ASTC_10X8_SRGB_BLOCK16);		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR
		case EF_ASTC_10x10_SRGB_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ASTC_10X10_SRGB_BLOCK16);		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR
		case EF_ASTC_12x10_SRGB_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ASTC_12X10_SRGB_BLOCK16);		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR
		case EF_ASTC_12x12_SRGB_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ASTC_12X12_SRGB_BLOCK16);		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR

		// packed formats
		case EF_E5B9G9R9_UFLOAT_PACK32: return getTranslatedFinalFormat(FORMAT_RGB9E5_UFLOAT_PACK32);			//GL_RGB9_E5
		case EF_B10G11R11_UFLOAT_PACK32: return getTranslatedFinalFormat(FORMAT_RG11B10_UFLOAT_PACK32);			//GL_R11F_G11F_B10F
		case EF_R5G6B5_UNORM_PACK16: return getTranslatedFinalFormat(FORMAT_R5G6B5_UNORM_PACK16);			//GL_RGB565
		case EF_R5G5B5A1_UNORM_PACK16: return getTranslatedFinalFormat(FORMAT_RGB5A1_UNORM_PACK16);			//GL_RGB5_A1
		case EF_R4G4B4A4_UNORM_PACK16: return getTranslatedFinalFormat(FORMAT_RGBA4_UNORM_PACK16);          //GL_RGBA4
		case EF_R4G4_UNORM_PACK8: return getTranslatedFinalFormat(FORMAT_RG4_UNORM_PACK8);

		// depth formats
		case EF_D16_UNORM: return getTranslatedFinalFormat(FORMAT_D16_UNORM_PACK16);				//GL_DEPTH_COMPONENT16
		case EF_X8_D24_UNORM_PACK32: return getTranslatedFinalFormat(FORMAT_D24_UNORM_S8_UINT_PACK32);				//GL_DEPTH_COMPONENT24
		case EF_D16_UNORM_S8_UINT: return getTranslatedFinalFormat(FORMAT_D16_UNORM_S8_UINT_PACK32);
		case EF_D24_UNORM_S8_UINT: return getTranslatedFinalFormat(FORMAT_D24_UNORM_S8_UINT_PACK32);			//GL_DEPTH24_STENCIL8
		case EF_D32_SFLOAT_S8_UINT: return getTranslatedFinalFormat(FORMAT_D32_SFLOAT_S8_UINT_PACK64);				//GL_DEPTH_COMPONENT32
		case EF_D32_SFLOAT: return getTranslatedFinalFormat(FORMAT_D32_SFLOAT_PACK32);				//GL_DEPTH_COMPONENT32F
		case EF_S8_UINT: return getTranslatedFinalFormat(FORMAT_S8_UINT_PACK8);

		// compressed formats
		case EF_PVRTC1_4BPP_UNORM_BLOCK_IMG: return getTranslatedFinalFormat(FORMAT_RGBA_PVRTC1_8X8_UNORM_BLOCK32);		//GL_COMPRESSED_RGBA_PVRTC_4BPPV1_IMG
		case EF_PVRTC1_2BPP_UNORM_BLOCK_IMG: return getTranslatedFinalFormat(FORMAT_RGBA_PVRTC1_8X8_UNORM_BLOCK32);		//GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG
		case EF_ETC2_R8G8B8_UNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_RGB_ETC2_UNORM_BLOCK8);					//GL_COMPRESSED_RGB8_ETC2
		case EF_ETC2_R8G8B8A8_UNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ETC2_UNORM_BLOCK8);				//GL_COMPRESSED_RGBA8_ETC2_EAC
		case EF_EAC_R11_UNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_R_EAC_UNORM_BLOCK8);						//GL_COMPRESSED_R11_EAC
		case EF_EAC_R11_SNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_R_EAC_SNORM_BLOCK8);						//GL_COMPRESSED_SIGNED_R11_EAC
		case EF_EAC_R11G11_UNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_RG_EAC_UNORM_BLOCK16);					//GL_COMPRESSED_RG11_EAC
		case EF_EAC_R11G11_SNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_RG_EAC_SNORM_BLOCK16);					//GL_COMPRESSED_SIGNED_RG11_EAC

		case EF_BC1_RGB_UNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_RGB_DXT1_UNORM_BLOCK8);						//GL_COMPRESSED_RGB_S3TC_DXT1_EXT
		case EF_BC1_RGBA_UNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_DXT1_UNORM_BLOCK8);					//GL_COMPRESSED_RGBA_S3TC_DXT1_EXT
		case EF_BC2_UNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_DXT3_UNORM_BLOCK16);					//GL_COMPRESSED_RGBA_S3TC_DXT3_EXT
		case EF_BC3_UNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_DXT5_UNORM_BLOCK16);					//GL_COMPRESSED_RGBA_S3TC_DXT5_EXT
		case EF_BC4_UNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_R_ATI1N_UNORM_BLOCK8);				//GL_COMPRESSED_RED_RGTC1
		case EF_BC4_SNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_R_ATI1N_SNORM_BLOCK8);				//GL_COMPRESSED_SIGNED_RED_RGTC1
		case EF_BC5_UNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_RG_ATI2N_UNORM_BLOCK16);				//GL_COMPRESSED_RG_RGTC2
		case EF_BC5_SNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_RG_ATI2N_SNORM_BLOCK16);				//GL_COMPRESSED_SIGNED_RG_RGTC2
		case EF_BC6H_UFLOAT_BLOCK: return getTranslatedFinalFormat(FORMAT_RGB_BP_UFLOAT_BLOCK16);		//GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT
		case EF_BC6H_SFLOAT_BLOCK: return getTranslatedFinalFormat(FORMAT_RGB_BP_SFLOAT_BLOCK16);			//GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT
		case EF_BC7_UNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_RGB_BP_UFLOAT_BLOCK16 /*there should be BP_UNORM, but no provided for RGB*/);					//GL_COMPRESSED_RGBA_BPTC_UNORM

		case EF_ASTC_4x4_UNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ASTC_4X4_UNORM_BLOCK16);				//GL_COMPRESSED_RGBA_ASTC_4x4_KHR
		case EF_ASTC_5x4_UNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ASTC_5X4_UNORM_BLOCK16);				//GL_COMPRESSED_RGBA_ASTC_5x4_KHR
		case EF_ASTC_5x5_UNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ASTC_5X5_UNORM_BLOCK16);				//GL_COMPRESSED_RGBA_ASTC_5x5_KHR
		case EF_ASTC_6x5_UNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ASTC_6X5_UNORM_BLOCK16);				//GL_COMPRESSED_RGBA_ASTC_6x5_KHR
		case EF_ASTC_6x6_UNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ASTC_6X6_UNORM_BLOCK16);				//GL_COMPRESSED_RGBA_ASTC_6x6_KHR
		case EF_ASTC_8x5_UNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ASTC_8X5_UNORM_BLOCK16);				//GL_COMPRESSED_RGBA_ASTC_8x5_KHR
		case EF_ASTC_8x6_UNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ASTC_8X6_UNORM_BLOCK16);				//GL_COMPRESSED_RGBA_ASTC_8x6_KHR
		case EF_ASTC_8x8_UNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ASTC_8X8_UNORM_BLOCK16);				//GL_COMPRESSED_RGBA_ASTC_8x8_KHR
		case EF_ASTC_10x5_UNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ASTC_10X5_UNORM_BLOCK16); 				//GL_COMPRESSED_RGBA_ASTC_10x5_KHR
		case EF_ASTC_10x6_UNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ASTC_10X6_UNORM_BLOCK16);				//GL_COMPRESSED_RGBA_ASTC_10x6_KHR
		case EF_ASTC_10x8_UNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ASTC_10X8_UNORM_BLOCK16);				//GL_COMPRESSED_RGBA_ASTC_10x8_KHR
		case EF_ASTC_10x10_UNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ASTC_10X10_UNORM_BLOCK16);				//GL_COMPRESSED_RGBA_ASTC_10x10_KHR
		case EF_ASTC_12x10_UNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ASTC_12X10_UNORM_BLOCK16);				//GL_COMPRESSED_RGBA_ASTC_12x10_KHR
		case EF_ASTC_12x12_UNORM_BLOCK: return getTranslatedFinalFormat(FORMAT_RGBA_ASTC_12X12_UNORM_BLOCK16);				//GL_COMPRESSED_RGBA_ASTC_12x12_KHR
			
		default: return getTranslatedFinalFormat();
		}
	}
}
}

#endif // _NBL_COMPILE_WITH_GLI_WRITER_
