// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>

#include "nbl/asset/filters/CRegionBlockFunctorFilter.h"

#include "CImageWriterOpenEXR.h"

#ifdef _NBL_COMPILE_WITH_OPENEXR_WRITER_


#include "ImfOutputFile.h"
#include "ImfChannelList.h"
#include "ImfChannelListAttribute.h"
#include "ImfStringAttribute.h"
#include "ImfMatrixAttribute.h"
#include "ImfArray.h"

#include "ImfFrameBuffer.h"
#include "ImfHeader.h"

#include "ImfNamespace.h"

namespace IMF = Imf;
namespace IMATH = Imath;

using namespace nbl;
using namespace nbl::asset;

using namespace IMF;
using namespace IMATH;

namespace nbl::asset::impl
{
	class nblOStream : public IMF::OStream
	{
	public:
		nblOStream(system::IFile* _nblFile)
			: IMF::OStream(getFileName(_nblFile).c_str()), nblFile(_nblFile) {}
		virtual ~nblOStream() {}

		//----------------------------------------------------------
		// Write to the stream:
		//
		// write(c,n) takes n bytes from array c, and stores them
		// in the stream.  If an I/O error occurs, write(c,n) throws
		// an exception.
		//----------------------------------------------------------

		virtual void write(const char c[/*n*/], int n) override
		{
			system::IFile::success_t success;
			nblFile->write(success, c, fileOffset, n);
			fileOffset += success.getBytesProcessed();
		}

		//---------------------------------------------------------
		// Get the current writing position, in bytes from the
		// beginning of the file.  If the next call to write() will
		// start writing at the beginning of the file, tellp()
		// returns 0.
		//---------------------------------------------------------

		virtual uint64_t tellp() override
		{
			return static_cast<uint64_t>(fileOffset);
		}

		//-------------------------------------------
		// Set the current writing position.
		// After calling seekp(i), tellp() returns i.
		//-------------------------------------------

		virtual void seekp(uint64_t pos) override
		{
			fileOffset = static_cast<decltype(fileOffset)>(pos);
		}

		void resetFileOffset()
		{
			fileOffset = 0u;
		}

	private:
		const std::string getFileName(system::IFile* _nblFile)
		{
			std::filesystem::path filename, extension;
			core::splitFilename(_nblFile->getFileName(), nullptr, &filename, &extension);
			return filename.string() + extension.string();
		}

		system::IFile* nblFile;
		size_t fileOffset = {};
	};
}

constexpr uint8_t availableChannels = 4;

template<typename ilmType>
bool createAndWriteImage(std::array<ilmType*, availableChannels>& pixelsArrayIlm, const asset::ICPUImage* image, system::IFile* _file)
{
	const auto& creationParams = image->getCreationParameters();
	auto getIlmType = [&creationParams]()
	{
		if (creationParams.format == EF_R16G16B16A16_SFLOAT)
			return PixelType::HALF;
		else if (creationParams.format == EF_R32G32B32A32_SFLOAT)
			return PixelType::FLOAT;
		else if (creationParams.format == EF_R32G32B32A32_UINT)
			return PixelType::UINT;
		else
			return PixelType::NUM_PIXELTYPES;
	};

	const auto width = creationParams.extent.width;
	const auto height = creationParams.extent.height;
	Header header(width, height);
	const PixelType pixelType = getIlmType();
	FrameBuffer frameBuffer;

	if (pixelType == PixelType::NUM_PIXELTYPES || creationParams.type != IImage::E_TYPE::ET_2D)
		return false;

	for (auto& channelPixelsPtr : pixelsArrayIlm)
		channelPixelsPtr = _NBL_NEW_ARRAY(ilmType, width * height);

	const auto* data = reinterpret_cast<const uint8_t*>(image->getBuffer()->getPointer());
	// have to use `std::function` cause MSVC is borderline retarded and feel the need to instantiate separate Lambda types for each reference!?
	auto writeTexel = std::function([&creationParams, &data, &pixelsArrayIlm](uint32_t ptrOffset, const core::vectorSIMDu32& texelCoord) -> void
		{
			assert(texelCoord.w == 0u && texelCoord.z == 0u);

			const uint8_t* texelPtr = data + ptrOffset;
			const uint64_t ptrStyleIlmShiftToDataChannelPixel = (texelCoord.y * creationParams.extent.width) + texelCoord.x;

			for (uint8_t channelIndex = 0; channelIndex < availableChannels; ++channelIndex)
			{
				ilmType channelPixel = *(reinterpret_cast<const ilmType*>(texelPtr) + channelIndex);
				*(pixelsArrayIlm[channelIndex] + ptrStyleIlmShiftToDataChannelPixel) = channelPixel;
			}
		});

	using StreamToEXR = CRegionBlockFunctorFilter<decltype(writeTexel), true>;
	typename StreamToEXR::state_type state(writeTexel, image, nullptr);
	for (const auto& region : image->getRegions())
	{
		if (region.imageSubresource.mipLevel || region.imageSubresource.baseArrayLayer)
			continue;

		state.regionIterator = &region;
		StreamToEXR::execute(core::execution::par_unseq, &state);
	}

	constexpr std::array<const char*, availableChannels> rgbaSignatureAsText = { "R", "G", "B", "A" };
	for (uint8_t channel = 0; channel < rgbaSignatureAsText.size(); ++channel)
	{
		auto rowPitch = sizeof(*pixelsArrayIlm[channel]) * width;

		header.channels().insert(rgbaSignatureAsText[channel], Channel(pixelType));
		frameBuffer.insert
		(
			rgbaSignatureAsText[channel],                                                                // name
			Slice(pixelType,                                                                             // type
				(char*)pixelsArrayIlm[channel],                                                             // base
				sizeof(*pixelsArrayIlm[channel]) * 1,                                                        // xStride
				rowPitch)																					 // yStride
		);
	}

	IMF::OStream* nblOStream = _NBL_NEW(asset::impl::nblOStream, _file);
	{ // brackets are needed because of OutputFile's destructor
		OutputFile file(*nblOStream, header);
		file.setFrameBuffer(frameBuffer);
		file.writePixels(height);
	}

	for (auto channelPixelsPtr : pixelsArrayIlm)
		_NBL_DELETE_ARRAY(channelPixelsPtr, width * height);
	_NBL_DELETE(nblOStream);

	return true;
}

bool CImageWriterOpenEXR::writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
{
	if (!_override)
		getDefaultOverride(_override);

	SAssetWriteContext ctx{ _params, _file };

	const ICPUImageView* imageView = IAsset::castDown<const ICPUImageView>(_params.rootAsset);
	const auto& viewParams = imageView->getCreationParameters();
	if (viewParams.image->missingContent())
		return false;

	core::smart_refctd_ptr<ICPUImage> imageSmart;
	if (asset::isIntegerFormat(viewParams.format))
		imageSmart = IImageAssetHandlerBase::createImageDataForCommonWriting<EF_R32G32B32A32_UINT>(imageView,_params.logger);
	else
	{
		bool halfFloat = true;
		for (auto ch=0; ch<4; ch++)
		if (getFormatMaxValue<hlsl::float64_t>(viewParams.format,ch)>getFormatMaxValue<hlsl::float64_t>(EF_R16G16B16A16_SFLOAT,ch))
		{
			halfFloat = false;
			break;
		}
		if (halfFloat)
			imageSmart = IImageAssetHandlerBase::createImageDataForCommonWriting<EF_R16G16B16A16_SFLOAT>(imageView,_params.logger);
		else
			imageSmart = IImageAssetHandlerBase::createImageDataForCommonWriting<EF_R32G32B32A32_SFLOAT>(imageView,_params.logger);
	}

	system::IFile* file = _override->getOutputFile(_file,ctx,{imageView,0u});
	if (!file)
		return false;

	return writeImageBinary(file,imageSmart.get());
}

bool CImageWriterOpenEXR::writeImageBinary(system::IFile* file, const asset::ICPUImage* image)
{
	const auto& params = image->getCreationParameters();

	std::array<half*, availableChannels> halfPixelMapArray = { nullptr, nullptr, nullptr, nullptr };
	std::array<float*, availableChannels> fullFloatPixelMapArray = { nullptr, nullptr, nullptr, nullptr };
	std::array<uint32_t*, availableChannels> uint32_tPixelMapArray = { nullptr, nullptr, nullptr, nullptr };

	if (params.format == EF_R16G16B16A16_SFLOAT)
		createAndWriteImage(halfPixelMapArray, image, file);
	else if (params.format == EF_R32G32B32A32_SFLOAT)
		createAndWriteImage(fullFloatPixelMapArray, image, file);
	else if (params.format == EF_R32G32B32A32_UINT)
		createAndWriteImage(uint32_tPixelMapArray, image, file);

	return true;
}
#endif // _NBL_COMPILE_WITH_OPENEXR_WRITER_
