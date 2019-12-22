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

		bool CImageWriterOpenEXR::writeAsset(io::IWriteFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
		{
			if (!_override)
				getDefaultOverride(_override);

			SAssetWriteContext ctx{ _params, _file };

			const asset::ICPUImage* image =
			#ifndef _IRR_DEBUG
				static_cast<const asset::ICPUImage*>(_params.rootAsset);
			#else
				dynamic_cast<const asset::ICPUImage*>(_params.rootAsset);
			#endif
			assert(image);

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

			Header header(params.extent.width, params.extent.height);
			PixelType pixelType;
			FrameBuffer frameBuffer;

			OutputFile ilmFile(file->getFileName().c_str(), header);

			if (params.type == EF_R16G16B16A16_SFLOAT)
				pixelType = PixelType::HALF;
			else if (params.type == EF_R32G32B32A32_SFLOAT)
				pixelType = PixelType::FLOAT;
			else if (params.type == EF_R32G32B32A32_UINT)
				pixelType = PixelType::UINT;

			constexpr std::array<char*, 4> rgbaSignatureAsText = { "R", "G", "B", "A" };
			for (const auto& channelAsText : rgbaSignatureAsText)
				header.channels().insert(channelAsText, Channel(pixelType));

			auto byteSizeOfSingleChannelPixel = params.format == EF_R16G16B16A16_SFLOAT ? 2 : 4;

			for (uint8_t rgbaChannelIndex = 0; rgbaChannelIndex < rgbaSignatureAsText.size(); ++rgbaChannelIndex)
				frameBuffer.insert
				(
					rgbaSignatureAsText[rgbaChannelIndex],                                                       // name
					Slice(pixelType,                                                                             // type
					(char*) image->getBuffer()->getPointer(),                                                    // base
					byteSizeOfSingleChannelPixel * 1,                                                            // xStride
					byteSizeOfSingleChannelPixel * params.extent.width                                           // yStride
				));

			ilmFile.setFrameBuffer(frameBuffer);
			ilmFile.writePixels(params.extent.height);

			return true;
		}
	}
}

#endif // _IRR_COMPILE_WITH_OPENEXR_WRITER_