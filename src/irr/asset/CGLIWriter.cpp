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

#ifdef _IRR_COMPILE_WITH_GLI_WRITER_

#ifdef _IRR_COMPILE_WITH_GLI_
#include "gli/gli.hpp"
#else
#error "It requires GLI library"
#endif

namespace irr
{
	namespace asset
	{
		static inline std::pair<gli::texture::format_type, std::array<gli::gl::swizzle, 4>> getTranslatedIRRFormat(const IImageView<ICPUImage>::SCreationParams& params);

		template<typename aType>
		static inline aType getSingleChannel(const void* data)
		{
			return *(reinterpret_cast<const aType*>(data));
		}

		static inline bool performSavingAsIWriteFile(gli::texture& texture, irr::io::IWriteFile* file);

		bool CGLIWriter::writeAsset(io::IWriteFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
		{
			if (!_override)
				getDefaultOverride(_override);

			SAssetWriteContext ctx{ _params, _file };

			const asset::ICPUImageView* imageView = IAsset::castDown<ICPUImageView>(_params.rootAsset);

			if (!imageView)
				return false;

			io::IWriteFile* file = _override->getOutputFile(_file, ctx, { imageView, 0u });

			if (!file)
				return false;

			return writeGLIFile(file, imageView);
		}

		bool CGLIWriter::writeGLIFile(io::IWriteFile* file, const asset::ICPUImageView* imageView)
		{
			os::Printer::log("WRITING GLI: writing the file", file->getFileName().c_str(), ELL_INFORMATION);

			const auto& imageViewInfo = imageView->getCreationParameters();
			const auto& imageInfo = imageViewInfo.image->getCreationParameters();
			const auto& image = imageViewInfo.image;

			if (image->getRegions().length() == 0)
			{
				os::Printer::log("WRITING GLI: there is a lack of regions!", file->getFileName().c_str(), ELL_INFORMATION);
				return false;
			}
			if (isBlockCompressionFormat(imageInfo.format))
			{
				os::Printer::log("WRITING GLI: the writer doesn't support any compression format!", file->getFileName().c_str(), ELL_INFORMATION);
				return false;
			}
				

			const bool isItACubemap = doesItHaveFaces(imageViewInfo.viewType);
			const bool layersFlag = doesItHaveLayers(imageViewInfo.viewType);

			const bool floatingPointFlag = isFloatingPointFormat(imageInfo.format);
			const bool integerFlag = isIntegerFormat(imageInfo.format);
			const bool signedTypeFlag = isSignedFormat(imageInfo.format);
		
			const auto texelBlockByteSize = asset::getTexelOrBlockBytesize(imageInfo.format);
			const auto channelsAmount = getFormatChannelCount(imageInfo.format);
			const auto singleChannelByteSize = texelBlockByteSize / channelsAmount;
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

			auto gliFormatAndSwizzles = getTranslatedIRRFormat(imageViewInfo);
			gli::target gliTarget = getTarget();
			gli::extent3d gliExtent3d = {imageInfo.extent.width, imageInfo.extent.height, imageInfo.extent.depth};
			size_t gliLevels = imageInfo.mipLevels;
			std::pair<size_t, size_t> layersAndFacesAmount = getFacesAndLayersAmount();

			gli::texture texture(gliTarget, gliFormatAndSwizzles.first, gliExtent3d, layersAndFacesAmount.first, layersAndFacesAmount.second, gliLevels, gli::texture::swizzles_type{ gliFormatAndSwizzles.second[0], gliFormatAndSwizzles.second[1], gliFormatAndSwizzles.second[2], gliFormatAndSwizzles.second[3] });

			auto getFullSizeOfLayer = [&](const uint16_t mipLevel) -> uint64_t
			{
				auto region = image->getRegions().begin() + mipLevel;
				const auto width = region->bufferRowLength == 0 ? region->imageExtent.width : region->bufferRowLength;
				const auto height = region->bufferImageHeight == 0 ? region->imageExtent.height : region->bufferImageHeight;
				const auto depth = region->imageExtent.depth;
				return width * texelBlockByteSize * height * depth;
			};

			for (auto region = image->getRegions().begin(); region != image->getRegions().end(); ++region)
			{
				const auto ptrBeginningOfRegion = data + region->bufferOffset;
				const auto layerSize = getFullSizeOfLayer(region->imageSubresource.mipLevel);
				const auto textureGliImgHeight = texture.extent(region->imageSubresource.mipLevel).y;
				const auto textureGliStride = texture.extent(region->imageSubresource.mipLevel).x;
				const auto textureGliStrideInBytes = textureGliStride * sizeof(singleChannelByteSize);
				const auto imgBufferWidth = region->bufferRowLength > 0 ? region->bufferRowLength : region->imageExtent.width;
				const auto imgBufferWidthInBytes = imgBufferWidth * sizeof(singleChannelByteSize);
				const auto imgBufferHeight = region->bufferImageHeight > 0 ? region->bufferImageHeight : region->imageExtent.height;
			
				for (uint16_t layer = 0; layer < imageInfo.arrayLayers; ++layer)
				{
					const uint16_t gliLayer = layersFlag ? layer / 6 : 0;
					const uint16_t gliFace = isItACubemap ? layer % 6 : 0;

					const auto layerData = texture.data(gliLayer, gliFace, region->imageSubresource.mipLevel);
					const auto sourceData = ptrBeginningOfRegion + (layer * layerSize);

					for (uint32_t yPos = 0; yPos < textureGliImgHeight; ++yPos)
						memcpy
						(
							reinterpret_cast<uint8_t*>(layerData) + (yPos * textureGliStrideInBytes),
							sourceData + (yPos * imgBufferWidthInBytes),
							imgBufferWidth
						);
				}	
			}

			return performSavingAsIWriteFile(texture, file);
		}

		bool performSavingAsIWriteFile(gli::texture& texture, irr::io::IWriteFile* file)
		{
			if (texture.empty())
				return false;

			const auto fileName = std::string(file->getFileName().c_str());
			std::vector<char> memory;
			bool properlyStatus;

			if (fileName.rfind(".dds") != std::string::npos)
				properlyStatus = save_dds(texture, memory);
			if (fileName.rfind(".kmg") != std::string::npos)
				properlyStatus = save_kmg(texture, memory);
			if (fileName.rfind(".ktx") != std::string::npos)
				properlyStatus = save_ktx(texture, memory);

			if (properlyStatus)
			{
				file->write(memory.data(), memory.size());
				return true;
			}
			else
			{
				os::Printer::log("WRITING GLI: failed to save the file", file->getFileName().c_str(), ELL_ERROR);
				return false;
			}
		}

		bool CGLIWriter::doesItHaveFaces(const IImageView<ICPUImage>::E_TYPE& type)
		{
			switch (type)
			{
				case ICPUImageView::ET_CUBE_MAP: return true;
				case ICPUImageView::ET_CUBE_MAP_ARRAY: return true;
				default: return false;
			}
		}

		bool CGLIWriter::doesItHaveLayers(const IImageView<ICPUImage>::E_TYPE& type)
		{
			switch (type)
			{
				case ICPUImageView::ET_1D_ARRAY: return true;
				case ICPUImageView::ET_2D_ARRAY: return true;
				case ICPUImageView::ET_CUBE_MAP_ARRAY: return true;
				default: return false;
			}
		}

		inline std::pair<gli::texture::format_type, std::array<gli::gl::swizzle, 4>> getTranslatedIRRFormat(const IImageView<ICPUImage>::SCreationParams& params)
		{
			using namespace gli;
			std::array<gli::gl::swizzle, 4> compomentMapping;

			static const core::unordered_map<ICPUImageView::SComponentMapping::E_SWIZZLE, gli::gl::swizzle> swizzlesMappingAPI =
			{
				std::make_pair(ICPUImageView::SComponentMapping::ES_R, gl::SWIZZLE_RED),
				std::make_pair(ICPUImageView::SComponentMapping::ES_G, gl::SWIZZLE_GREEN),
				std::make_pair(ICPUImageView::SComponentMapping::ES_B, gl::SWIZZLE_BLUE),
				std::make_pair(ICPUImageView::SComponentMapping::ES_A, gl::SWIZZLE_ALPHA),
				std::make_pair(ICPUImageView::SComponentMapping::ES_ONE, gl::SWIZZLE_ONE),
				std::make_pair(ICPUImageView::SComponentMapping::ES_ZERO, gl::SWIZZLE_ZERO)
			};

			auto getMappedSwizzle = [&](const ICPUImageView::SComponentMapping::E_SWIZZLE& currentSwizzleToCheck)
			{
				for (auto& mappedSwizzle : swizzlesMappingAPI)
					if (currentSwizzleToCheck == mappedSwizzle.first)
						return mappedSwizzle.second;
			};

			compomentMapping[0] = getMappedSwizzle(params.components.r);
			compomentMapping[1] = getMappedSwizzle(params.components.g);
			compomentMapping[2] = getMappedSwizzle(params.components.b);
			compomentMapping[3] = getMappedSwizzle(params.components.a);

			auto getTranslatedFinalFormat = [&](const gli::texture::format_type& format = FORMAT_UNDEFINED)
			{
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

			// Floating formats
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

			// Compressed formats
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

#endif // _IRR_COMPILE_WITH_GLI_WRITER_