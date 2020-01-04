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

			gli::target gliTarget = getTarget();
			gli::format gliFormat;						 // TODO ! ! ! format + swizzle
			gli::extent3d gliExtent3d = {imageInfo.extent.width, imageInfo.extent.height, imageInfo.extent.depth};
			size_t gliLevels = imageInfo.mipLevels;
			std::pair<size_t, size_t> layersAndFacesAmount = getFacesAndLayersAmount();

			gli::texture texture(gliTarget, gliFormat, gliExtent3d, layersAndFacesAmount.first, layersAndFacesAmount.second, gliLevels);

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
	}
}

#endif // _IRR_COMPILE_WITH_GLI_WRITER_