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
		inline std::pair<gli::texture::format_type, std::array<gli::gl::swizzle, 4>> getTranslatedIRRFormat(const IImageView<ICPUImage>::SCreationParams& params);

		template<typename aType>
		aType getSingleChannel(const void* data)
		{
			return *(reinterpret_cast<const aType*>(data));
		}

		bool CGLIWriter::writeAsset(io::IWriteFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
		{
			if (!_override)
				getDefaultOverride(_override);

			SAssetWriteContext ctx{ _params, _file };

			const asset::ICPUImageView* imageView = IAsset::castDown<ICPUImageView>(_params.rootAsset);

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

			const bool facesFlag = doesItHaveFaces(imageViewInfo.viewType);
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
				const auto sizeOfRegion = image->getRegions().length;

				if (layersFlag)
					layers = ((sizeOfRegion - 1) % 6) + 1;
				else
					layers = 0;
				if (facesFlag)
					faces = ((sizeOfRegion - 1) / 6) + 1;
				else
					faces = 0;

				return std::make_pair(layers, faces);
			};

			auto gliFormatAndSwizzles = getTranslatedIRRFormat(imageViewInfo);
			gli::target gliTarget = getTarget();
			gli::extent3d gliExtent3d = {imageInfo.extent.width, imageInfo.extent.height, imageInfo.extent.depth};
			size_t gliLevels = imageInfo.mipLevels;
			std::pair<size_t, size_t> layersAndFacesAmount = getFacesAndLayersAmount();

			gli::texture texture(gliTarget, gliFormatAndSwizzles.first, gliExtent3d, layersAndFacesAmount.first, layersAndFacesAmount.second, gliLevels);

			const auto begginingOfRegion = image->getRegions().begin();
			for (uint16_t layer = 0; layer < imageInfo.arrayLayers; ++layer)
			{
				const uint16_t gliLayer = layersFlag ? layer % 6 : 0;
				const uint16_t gliFace = facesFlag ? layer / 6 : 0;

				for (uint16_t mipLevel = 0; mipLevel < imageInfo.mipLevels; ++mipLevel)
				{
					const auto region = (begginingOfRegion + mipLevel);
					const auto width = region->bufferRowLength == 0 ? region->imageExtent.width : region->bufferRowLength;
					const auto height = region->bufferImageHeight == 0 ? region->imageExtent.height : region->bufferImageHeight;
					const auto depth = region->imageExtent.depth;

					for (uint64_t xPos = 0; xPos < width; ++xPos)
					{
						for (uint64_t yPos = 0; yPos < height; ++yPos)
						{
							for (uint64_t zPos = 0; zPos < depth; ++zPos)
							{
								const auto texelStridePtr = data + ((zPos * height + yPos) * width + xPos) * texelBlockByteSize + region->bufferOffset;
								for (uint8_t channelIndex = 0; channelIndex < channelsAmount; ++channelIndex)
								{
									if (floatingPointFlag)
										texture.store({ xPos, yPos, zPos }, gliLayer, gliFace, mipLevel, getSingleChannel<float>(texelStridePtr + (channelIndex * singleChannelByteSize)));
									else if(integerFlag)
										if(signedTypeFlag)
											texture.store({ xPos, yPos, zPos }, gliLayer, gliFace, mipLevel, getSingleChannel<int64_t>(texelStridePtr + (channelIndex * singleChannelByteSize)));
										else
											texture.store({ xPos, yPos, zPos }, gliLayer, gliFace, mipLevel, getSingleChannel<uint64_t>(texelStridePtr + (channelIndex * singleChannelByteSize)));
								}		
							}
						}
					}
				}
			}

			return gli::save(texture, file->getFileName().c_str());
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

			auto getTranslatedFinalFormat = [&](const gli::texture::format_type& format = FORMAT_UNDEFINED)
			{
				return std::make_pair(format, std::array<gli::gl::swizzle, 4>{gl::SWIZZLE_RED, gl::SWIZZLE_GREEN, gl::SWIZZLE_BLUE, gl::SWIZZLE_ALPHA}); // TODO swizzles
			};

			// TODO - fill formats
			switch (params.format)
			{
				/*
			case EF_R8G8B8_UNORM: return getTranslatedFinalFormat(gl::INTERNAL_RGB_UNORM);			//GL_RGB
			case EF_B8G8R8_UNORM: return getTranslatedFinalFormat(gl::INTERNAL_BGR_UNORM);			//GL_BGR
			case EF_R8G8B8A8_UNORM: return getTranslatedFinalFormat(gl::INTERNAL_RGBA_UNORM);		//GL_RGBA
			case EF_B8G8R8A8_UNORM: return getTranslatedFinalFormat(gl::INTERNAL_BGRA_UNORM);		//GL_BGRA

			// unorm formats
			case EF_R8_UNORM: return getTranslatedFinalFormat(gl::INTERNAL_R8_UNORM);			//GL_R8
			case EF_R8G8_UNORM: return getTranslatedFinalFormat(gl::INTERNAL_RG8_UNORM);		//GL_RG8
			//case EF_R8G8B8_UNORM: return getTranslatedFinalFormat(gl::INTERNAL_RGB8_UNORM);		//GL_RGB8
			//case EF_R8G8B8A8_UNORM: return getTranslatedFinalFormat(gl::INTERNAL_RGBA8_UNORM);		//GL_RGBA8

			case EF_R16_UNORM: return getTranslatedFinalFormat(gl::INTERNAL_R16_UNORM);		//GL_R16
			case EF_R16G16_UNORM: return getTranslatedFinalFormat(gl::INTERNAL_RG16_UNORM);		//GL_RG16
			case EF_R16G16B16_UNORM: return getTranslatedFinalFormat(gl::INTERNAL_RGB16_UNORM);		//GL_RGB16
			case EF_R16G16B16A16_UNORM: return getTranslatedFinalFormat(gl::INTERNAL_RGBA16_UNORM);		//GL_RGBA16

			case EF_A2R10G10B10_UNORM_PACK32: return getTranslatedFinalFormat(gl::INTERNAL_RGB10A2_UNORM);	//GL_RGB10_A2
			//case EF_A2R10G10B10_UNORM_PACK32: return getTranslatedFinalFormat(gl::INTERNAL_RGB10A2_SNORM_EXT);

				// snorm formats
			case EF_R8_SNORM: return getTranslatedFinalFormat(gl::INTERNAL_R8_SNORM);			//GL_R8_SNORM
			case EF_R8G8_SNORM: return getTranslatedFinalFormat(gl::INTERNAL_RG8_SNORM);		//GL_RG8_SNORM
			case EF_R8G8B8_SNORM: return getTranslatedFinalFormat(gl::INTERNAL_RGB8_SNORM);		//GL_RGB8_SNORM
			case EF_R8G8B8A8_SNORM: return getTranslatedFinalFormat(gl::INTERNAL_RGBA8_SNORM);		//GL_RGBA8_SNORM

			case EF_R16_SNORM: return getTranslatedFinalFormat(gl::INTERNAL_R16_SNORM);		//GL_R16_SNORM
			case EF_R16G16_SNORM: return getTranslatedFinalFormat(gl::INTERNAL_RG16_SNORM);		//GL_RG16_SNORM
			case EF_R16G16B16_SNORM: return getTranslatedFinalFormat(gl::INTERNAL_RGB16_SNORM);		//GL_RGB16_SNORM
			case EF_R16G16B16A16_SNORM: return getTranslatedFinalFormat(gl::INTERNAL_RGBA16_SNORM);		//GL_RGBA16_SNORM

			// unsigned integer formats
			case EF_R8_UINT: return getTranslatedFinalFormat(gl::INTERNAL_R8U);				//GL_R8UI
			case EF_R8G8_UINT: return getTranslatedFinalFormat(gl::INTERNAL_RG8U);				//GL_RG8UI
			case EF_R8G8B8_UINT: return getTranslatedFinalFormat(gl::INTERNAL_RGB8U);			//GL_RGB8UI
			case EF_R8G8B8A8_UINT: return getTranslatedFinalFormat(gl::INTERNAL_RGBA8U);			//GL_RGBA8UI

			case EF_R16_UINT: return getTranslatedFinalFormat(gl::INTERNAL_R16U);				//GL_R16UI
			case EF_R16G16_UINT: return getTranslatedFinalFormat(gl::INTERNAL_RG16U);			//GL_RG16UI
			case EF_R16G16B16_UINT: return getTranslatedFinalFormat(gl::INTERNAL_RGB16U);			//GL_RGB16UI
			case EF_R16G16B16A16_UINT: return getTranslatedFinalFormat(gl::INTERNAL_RGBA16U);			//GL_RGBA16UI

			case EF_R32_UINT: return getTranslatedFinalFormat(gl::INTERNAL_R32U);				//GL_R32UI
			case EF_R32G32_UINT: return getTranslatedFinalFormat(gl::INTERNAL_RG32U);			//GL_RG32UI
			case EF_R32G32B32_UINT: return getTranslatedFinalFormat(gl::INTERNAL_RGB32U);			//GL_RGB32UI
			case EF_R32G32B32A32_UINT: return getTranslatedFinalFormat(gl::INTERNAL_RGBA32U);			//GL_RGBA32UI

			case EF_A2R10G10B10_UINT_PACK32: return getTranslatedFinalFormat(gl::INTERNAL_RGB10A2U);			//GL_RGB10_A2UI
			case EF_A2R10G10B10_SINT_PACK32: return getTranslatedFinalFormat(gl::INTERNAL_RGB10A2I_EXT);	//GL_RGB10_A2I

			// signed integer formats
			case EF_R8_SINT: return getTranslatedFinalFormat(gl::INTERNAL_R8I);				//GL_R8I
			case EF_R8G8_SINT: return getTranslatedFinalFormat(gl::INTERNAL_RG8I);				//GL_RG8I
			case EF_R8G8B8_SINT: return getTranslatedFinalFormat(gl::INTERNAL_RGB8I);			//GL_RGB8I
			case EF_R8G8B8A8_SINT: return getTranslatedFinalFormat(gl::INTERNAL_RGBA8I);			//GL_RGBA8I

			case EF_R16_SINT: return getTranslatedFinalFormat(gl::INTERNAL_R16I);				//GL_R16I
			case EF_R16G16_SINT: return getTranslatedFinalFormat(gl::INTERNAL_RG16I);			//GL_RG16I
			case EF_R16G16B16_SINT: return getTranslatedFinalFormat(gl::INTERNAL_RGB16I);			//GL_RGB16I
			case EF_R16G16B16A16_SINT: return getTranslatedFinalFormat(gl::INTERNAL_RGBA16I);			//GL_RGBA16I

			case EF_R32_SINT: return getTranslatedFinalFormat(gl::INTERNAL_R32I);				//GL_R32I
			case EF_R32G32_SINT: return getTranslatedFinalFormat(gl::INTERNAL_RG32I);			//GL_RG32I
			case EF_R32G32B32_SINT: return getTranslatedFinalFormat(gl::INTERNAL_RGB32I);			//GL_RGB32I
			case EF_R32G32B32A32_SINT: return getTranslatedFinalFormat(gl::INTERNAL_RGBA32I);			//GL_RGBA32I

			// Floating formats
			case EF_R16_SFLOAT: return getTranslatedFinalFormat(gl::INTERNAL_R16F);				//GL_R16F
			case EF_R16G16_SFLOAT: return getTranslatedFinalFormat(gl::INTERNAL_RG16F);			//GL_RG16F
			case EF_R16G16B16_SFLOAT: return getTranslatedFinalFormat(gl::INTERNAL_RGB16F);			//GL_RGB16F
			case EF_R16G16B16A16_SFLOAT: return getTranslatedFinalFormat(gl::INTERNAL_RGBA16F);			//GL_RGBA16F

			case EF_R32_SFLOAT: return getTranslatedFinalFormat(gl::INTERNAL_R32F);				//GL_R32F
			case EF_R32G32_SFLOAT: return getTranslatedFinalFormat(gl::INTERNAL_RG32F);			//GL_RG32F
			case EF_R32G32B32_SFLOAT: return getTranslatedFinalFormat(gl::INTERNAL_RGB32F);			//GL_RGB32F
			case EF_R32G32B32A32_SFLOAT: return getTranslatedFinalFormat(gl::INTERNAL_RGBA32F);			//GL_RGBA32F

			case EF_R64_SFLOAT: return getTranslatedFinalFormat(gl::INTERNAL_R64F_EXT);			//GL_R64F
			case EF_R64G64_SFLOAT: return getTranslatedFinalFormat(gl::INTERNAL_RG64F_EXT);		//GL_RG64F
			case EF_R64G64B64_SFLOAT: return getTranslatedFinalFormat(gl::INTERNAL_RGB64F_EXT);		//GL_RGB64F
			case EF_R64G64B64A64_SFLOAT: return getTranslatedFinalFormat(gl::INTERNAL_RGBA64F_EXT);		//GL_RGBA64F

			// sRGB formats
			case EF_R8_SRGB: return getTranslatedFinalFormat(gl::INTERNAL_SR8);				//GL_SR8_EXT
			case EF_R8G8_SRGB: return getTranslatedFinalFormat(gl::INTERNAL_SRG8);				//GL_SRG8_EXT
			case EF_R8G8B8_SRGB: return getTranslatedFinalFormat(gl::INTERNAL_SRGB8);			//GL_SRGB8
			case EF_R8G8B8A8_SRGB: return getTranslatedFinalFormat(gl::INTERNAL_SRGB8_ALPHA8);		//GL_SRGB8_ALPHA8

			// Compressed formats
			case EF_ASTC_4x4_UNORM_BLOCK: return getTranslatedFinalFormat(gl::INTERNAL_RGBA_ASTC_4x4);				//GL_COMPRESSED_RGBA_ASTC_4x4_KHR
			case EF_ASTC_5x4_UNORM_BLOCK: return getTranslatedFinalFormat(gl::INTERNAL_RGBA_ASTC_5x4);				//GL_COMPRESSED_RGBA_ASTC_5x4_KHR
			case EF_ASTC_5x5_UNORM_BLOCK: return getTranslatedFinalFormat(gl::INTERNAL_RGBA_ASTC_5x5);				//GL_COMPRESSED_RGBA_ASTC_5x5_KHR
			case EF_ASTC_6x5_UNORM_BLOCK: return getTranslatedFinalFormat(gl::INTERNAL_RGBA_ASTC_6x5);				//GL_COMPRESSED_RGBA_ASTC_6x5_KHR
			case EF_ASTC_6x6_UNORM_BLOCK: return getTranslatedFinalFormat(gl::INTERNAL_RGBA_ASTC_6x6);				//GL_COMPRESSED_RGBA_ASTC_6x6_KHR
			case EF_ASTC_8x5_UNORM_BLOCK: return getTranslatedFinalFormat(gl::INTERNAL_RGBA_ASTC_8x5);				//GL_COMPRESSED_RGBA_ASTC_8x5_KHR
			case EF_ASTC_8x6_UNORM_BLOCK: return getTranslatedFinalFormat(gl::INTERNAL_RGBA_ASTC_8x6);				//GL_COMPRESSED_RGBA_ASTC_8x6_KHR
			case EF_ASTC_8x8_UNORM_BLOCK: return getTranslatedFinalFormat(gl::INTERNAL_RGBA_ASTC_8x8);				//GL_COMPRESSED_RGBA_ASTC_8x8_KHR
			case EF_ASTC_10x5_UNORM_BLOCK: return getTranslatedFinalFormat(gl::INTERNAL_RGBA_ASTC_10x5); 				//GL_COMPRESSED_RGBA_ASTC_10x5_KHR
			case EF_ASTC_10x6_UNORM_BLOCK: return getTranslatedFinalFormat(gl::INTERNAL_RGBA_ASTC_10x6);				//GL_COMPRESSED_RGBA_ASTC_10x6_KHR
			case EF_ASTC_10x8_UNORM_BLOCK: return getTranslatedFinalFormat(gl::INTERNAL_RGBA_ASTC_10x8);				//GL_COMPRESSED_RGBA_ASTC_10x8_KHR
			case EF_ASTC_10x10_UNORM_BLOCK: return getTranslatedFinalFormat(gl::INTERNAL_RGBA_ASTC_10x10);				//GL_COMPRESSED_RGBA_ASTC_10x10_KHR
			case EF_ASTC_12x10_UNORM_BLOCK: return getTranslatedFinalFormat(gl::INTERNAL_RGBA_ASTC_12x10);				//GL_COMPRESSED_RGBA_ASTC_12x10_KHR
			case EF_ASTC_12x12_UNORM_BLOCK: return getTranslatedFinalFormat(gl::INTERNAL_RGBA_ASTC_12x12);				//GL_COMPRESSED_RGBA_ASTC_12x12_KHR
			*/
			default: return getTranslatedFinalFormat();
			}
		}
	}
}

#endif // _IRR_COMPILE_WITH_GLI_WRITER_