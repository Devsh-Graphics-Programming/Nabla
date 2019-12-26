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

#include "CImageWriterOpenEXR.h"

#ifdef _IRR_COMPILE_WITH_OPENEXR_WRITER_

#include "openexr/IlmBase/Imath/ImathBox.h"
#include "openexr/OpenEXR/IlmImf/ImfOutputFile.h"
#include "openexr/OpenEXR/IlmImf/ImfChannelList.h"
#include "openexr/OpenEXR/IlmImf/ImfChannelListAttribute.h"
#include "openexr/OpenEXR/IlmImf/ImfStringAttribute.h"
#include "openexr/OpenEXR/IlmImf/ImfMatrixAttribute.h"
#include "openexr/OpenEXR/IlmImf/ImfArray.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>

#include "openexr/OpenEXR/IlmImf/ImfNamespace.h"
namespace IMF = OPENEXR_IMF_NAMESPACE;
namespace IMATH = IMATH_NAMESPACE;

namespace irr
{
	namespace asset
	{
		using namespace IMF;
		using namespace IMATH;

		constexpr uint8_t availableChannels = 4;

		template<typename ilmType>
		bool createAndWriteImage(std::array<ilmType*, availableChannels>& pixelsArrayIlm, const asset::ICPUImage* image, const char* fileName)
		{
			auto getIlmType = [&]()
			{
				if (image->getCreationParameters().format == EF_R16G16B16A16_SFLOAT)
					return PixelType::HALF;
				else if (image->getCreationParameters().format == EF_R32G32B32A32_SFLOAT)
					return PixelType::FLOAT;
				else if (image->getCreationParameters().format == EF_R32G32B32A32_UINT)
					return PixelType::UINT;
				else
					return PixelType::NUM_PIXELTYPES;
			};

			Header header(image->getCreationParameters().extent.width, image->getCreationParameters().extent.height);
			const PixelType pixelType = getIlmType();
			FrameBuffer frameBuffer;

			if (pixelType == PixelType::NUM_PIXELTYPES || image->getCreationParameters().type != IImage::E_TYPE::ET_2D)
				return false;

			uint64_t width = {};
			uint64_t height = {};
			std::vector<const IImage::SBufferCopy*> regionsToHandle;

			for (auto region = image->getRegions().begin(); region != image->getRegions().end(); ++region)
				if (region->imageSubresource.mipLevel == 0)
				{
					regionsToHandle.push_back(region);
					width += region->imageExtent.width;
					height += region->imageExtent.height;
				}

			for (auto& channelPixelsPtr : pixelsArrayIlm)
				channelPixelsPtr = _IRR_NEW_ARRAY(ilmType, width * height);

			auto data = image->getBuffer()->getPointer();
			for (auto region : regionsToHandle)
			{
				for (uint64_t yPos = region->imageOffset.y; yPos < region->imageOffset.y + region->imageExtent.height; ++yPos)
					for (uint64_t xPos = region->imageOffset.x; xPos < region->imageOffset.x + region->imageExtent.width; ++xPos)
					{
						const uint64_t ptrStyleEndShiftToImageDataPixel = (yPos * width * availableChannels) + (xPos * availableChannels);
						const uint64_t ptrStyleIlmShiftToDataChannelPixel = (yPos * width) + xPos;

						for (uint8_t channelIndex = 0; channelIndex < availableChannels; ++channelIndex)
						{
							ilmType channelPixel = *(reinterpret_cast<const ilmType*>(data) + ptrStyleEndShiftToImageDataPixel + channelIndex);
							*(pixelsArrayIlm[channelIndex] + ptrStyleIlmShiftToDataChannelPixel) = channelPixel;
						}
					}
			}

			constexpr std::array<char*, availableChannels> rgbaSignatureAsText = { "R", "G", "B", "A" };
			for (uint8_t channel = 0; channel < rgbaSignatureAsText.size(); ++channel)
			{
				header.channels().insert(rgbaSignatureAsText[channel], Channel(pixelType));
				frameBuffer.insert
				(
					rgbaSignatureAsText[channel],                                                                // name
					Slice(pixelType,                                                                             // type
					(char*) pixelsArrayIlm[channel],                                                             // base
					sizeof(*pixelsArrayIlm[channel]) * 1,                                                        // xStride
					sizeof(*pixelsArrayIlm[channel]) * width)                                                    // yStride
				);
			}

			OutputFile file(fileName, header);
			file.setFrameBuffer(frameBuffer);
			file.writePixels(height);

			for (auto channelPixelsPtr : pixelsArrayIlm)
				_IRR_DELETE_ARRAY(channelPixelsPtr, width * height);
		}

		bool CImageWriterOpenEXR::writeAsset(io::IWriteFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
		{
			if (!_override)
				getDefaultOverride(_override);

			SAssetWriteContext ctx{ _params, _file };

			const asset::ICPUImage* image = IAsset::castDown<ICPUImage>(_params.rootAsset);

			if (image->getBuffer()->isADummyObjectForCache())
				return false;

			io::IWriteFile* file = _override->getOutputFile(_file, ctx, { image, 0u });

			if (!file)
				return false;

			os::Printer::log("Writing OpenEXR image", file->getFileName().c_str());

			const asset::E_WRITER_FLAGS flags = _override->getAssetWritingFlags(ctx, image, 0u);
			if (flags & asset::EWF_BINARY)
				return writeImageBinary(file, image);
		}

		bool CImageWriterOpenEXR::writeImageBinary(io::IWriteFile* file, const asset::ICPUImage* image)
		{
			const auto& params = image->getCreationParameters();
			
			std::array<half*, availableChannels> halfPixelMapArray = {nullptr, nullptr, nullptr, nullptr};
			std::array<float*, availableChannels> fullFloatPixelMapArray = { nullptr, nullptr, nullptr, nullptr };
			std::array<uint32_t*, availableChannels> uint32_tPixelMapArray = { nullptr, nullptr, nullptr, nullptr };

			if (params.format == EF_R16G16B16A16_SFLOAT)
				createAndWriteImage(halfPixelMapArray, image, file->getFileName().c_str());
			else if (params.format == EF_R32G32B32A32_SFLOAT)
				createAndWriteImage(fullFloatPixelMapArray, image, file->getFileName().c_str());
			else if (params.format == EF_R32G32B32A32_UINT)
				createAndWriteImage(uint32_tPixelMapArray, image, file->getFileName().c_str());

			return true;
		}
	}
}

#endif // _IRR_COMPILE_WITH_OPENEXR_WRITER_