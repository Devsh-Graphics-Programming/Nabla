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

#include "CImageLoaderOpenEXR.h"

#ifdef _IRR_COMPILE_WITH_OPENEXR_LOADER_

#include "openexr/IlmBase/Imath/ImathBox.h"
#include "openexr/OpenEXR/IlmImf/ImfRgbaFile.h"
#include "openexr/OpenEXR/IlmImf/ImfInputFile.h"
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

		class SContext;
		bool readVersionField(io::IReadFile* _file, SContext& ctx);
		bool readHeader(const char fileName[], SContext& ctx);
		template<typename rgbaFormat>
		void readRgba(InputFile& file, std::array<Array2D<rgbaFormat>, 4>& pixelRgbaMapArray, int& width, int& height, E_FORMAT& format, const uint8_t& availableChannels);
		bool specifyIrrlichtEndFormat(E_FORMAT& format, const InputFile& file, bool& doesItSupportAlphaChannel);

		//! A helpful struct for handling OpenEXR layout
		/*
			The latest OpenEXR file consists of the following components:
			- magic number
			- version field
			- header
			- line offset table
			- scan line blocks
		*/
		struct SContext
		{
			constexpr static uint32_t magicNumber = 20000630ul; // 0x76, 0x2f, 0x31, 0x01

			struct VersionField
			{
				uint32_t mainDataRegisterField = 0ul;      // treated as 2 seperate bit fields, contains some usefull data
				uint8_t fileFormatVersionNumber = 0;	   // contains current OpenEXR version. It has to be 0 upon initialization!
				bool doesFileContainLongNames;			   // if set, the maximum length of attribute names, attribute type names and channel names is 255 bytes. Otherwise 31 bytes
				bool doesItSupportDeepData;		           // if set, there is at least one part which is not a regular scan line image or regular tiled image, so it is a deep format

				struct Compoment
				{
					enum CompomentType
					{
						SINGLE_PART_FILE,
						MULTI_PART_FILE
					};

					enum SinglePartFileCompoments
					{
						NONE,
						SCAN_LINES,
						TILES,
						SCAN_LINES_OR_TILES
					};

					CompomentType type;
					SinglePartFileCompoments singlePartFileCompomentSubTypes;

				} Compoment;

			} versionField;

			struct Attributes
			{
				// The header of every OpenEXR file must contain at least the following attributes
				//according to https://www.openexr.com/documentation/openexrfilelayout.pdf (page 8)
				const IMF::Channel* channels = nullptr;
				const IMF::Compression* compression = nullptr;
				const IMATH::Box2i* dataWindow = nullptr;
				const IMATH::Box2i* displayWindow = nullptr;
				const IMF::LineOrder* lineOrder = nullptr;
				const float* pixelAspectRatio = nullptr;
				const IMATH::V2f* screenWindowCenter = nullptr;
				const float* screenWindowWidth = nullptr;

				// These attributes are required in the header for all multi - part and /or deep data OpenEXR files
				const std::string* name = nullptr;
				const std::string* type = nullptr;
				const int* version = nullptr;
				const int* chunkCount = nullptr;

				// This attribute is required in the header for all files which contain deep data (deepscanline or deeptile)
				const int* maxSamplesPerPixel = nullptr;

				// This attribute is required in the header for all files which contain one or more tiles
				const IMF::TileDescription* tiles = nullptr;

				// This attribute can be used in the header for multi-part files
				const std::string* view = nullptr;

				// Others not required that can be used by metadata
				// - none at the moment

			} attributes;

			// core::smart_refctd_dynamic_array<uint32_t> offsetTable; 

			// scan line blocks
		};

		template<typename IlmF>
		void trackIrrFormatAndPerformDataAssignment(void* begginingOfImageDataBuffer, const uint64_t& endShiftToSpecifyADataPos, const IlmF& ilmPixelValueToAssignTo)
		{
			*(reinterpret_cast<IlmF*>(begginingOfImageDataBuffer) + endShiftToSpecifyADataPos) = ilmPixelValueToAssignTo;
		}

		asset::SAssetBundle CImageLoaderOpenEXR::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
		{
			if (!_file)
				return {};

			const auto& fileName = _file->getFileName().c_str();

			SContext ctx;
			InputFile file = fileName;

			if (!readVersionField(_file, ctx))
				return {};

			if (!readHeader(fileName, ctx))
				return {};

			constexpr uint8_t availableChannels = 4;
			std::array<Array2D<half>, availableChannels> halfPixelMapArray;
			std::array<Array2D<float>, availableChannels> fullFloatPixelMapArray;
			std::array<Array2D<uint32_t>, availableChannels> uint32_tPixelMapArray;

			int width;
			int height;
			bool doesItSupportAlphaChannel;

			ICPUImage::SCreationParams params;
			params.type = ICPUImage::ET_2D;;
			params.flags = static_cast<ICPUImage::E_CREATE_FLAGS>(0u);
			params.samples = ICPUImage::ESCF_1_BIT;
			params.extent.depth = 1u;
			params.mipLevels = 1u;
			params.arrayLayers = 1u;

			if (!specifyIrrlichtEndFormat(params.format, file, doesItSupportAlphaChannel))
				return {};

			if (params.format == EF_R16G16B16A16_SFLOAT)
				readRgba(file, halfPixelMapArray, width, height, params.format, availableChannels);
			else if(params.format == EF_R32G32B32A32_SFLOAT)
				readRgba(file, fullFloatPixelMapArray, width, height, params.format, availableChannels);
			else if (params.format == EF_R32G32B32A32_UINT)
				readRgba(file, uint32_tPixelMapArray, width, height, params.format, availableChannels);

			params.extent.width = width;
			params.extent.height = height;

			auto image = ICPUImage::create(std::move(params));

			const uint32_t texelFormatByteSize = getTexelOrBlockBytesize(image->getCreationParameters().format);
			auto texelBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(image->getImageDataSizeInBytes());
			auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1u);
			ICPUImage::SBufferCopy& region = regions->front();
			//region.imageSubresource.aspectMask = ...; // waits for Vulkan
			region.imageSubresource.mipLevel = 0u;
			region.imageSubresource.baseArrayLayer = 0u;
			region.imageSubresource.layerCount = 1u;
			region.bufferOffset = 0u;
			region.bufferRowLength = calcPitchInBlocks(width, texelFormatByteSize);
			region.bufferImageHeight = 0u;
			region.imageOffset = { 0u, 0u, 0u };
			region.imageExtent = image->getCreationParameters().extent;
		
			void* fetchedData = texelBuffer->getPointer();
			const auto pitch = region.bufferRowLength;

			for (uint64_t yPos = 0; yPos < height; ++yPos)
				for (uint64_t xPos = 0; xPos < width; ++xPos)
				{
					const uint64_t ptrStyleEndShiftToImageDataPixel = (yPos * pitch * availableChannels) + (xPos * availableChannels);

					for (uint8_t channelIndex = 0; channelIndex < availableChannels; ++channelIndex)
					{
						const auto& halfChannelElement =      (halfPixelMapArray[channelIndex])[yPos][xPos];
						const auto& fullFloatChannelElement = (fullFloatPixelMapArray[channelIndex])[yPos][xPos];
						const auto& uint32_tChannelElement =  (uint32_tPixelMapArray[channelIndex])[yPos][xPos];

						if (params.format == EF_R16G16B16A16_SFLOAT)
							trackIrrFormatAndPerformDataAssignment<half>(fetchedData, ptrStyleEndShiftToImageDataPixel + channelIndex, halfChannelElement);
						else if (params.format == EF_R32G32B32A32_SFLOAT)
							trackIrrFormatAndPerformDataAssignment<float>(fetchedData, ptrStyleEndShiftToImageDataPixel + channelIndex, fullFloatChannelElement);
						else if (params.format == EF_R32G32B32A32_UINT)
							trackIrrFormatAndPerformDataAssignment<uint32_t>(fetchedData, ptrStyleEndShiftToImageDataPixel + channelIndex, uint32_tChannelElement);
					}
				}

			image->setBufferAndRegions(std::move(texelBuffer), regions);

			return SAssetBundle({image});
		}

		bool CImageLoaderOpenEXR::isALoadableFileFormat(io::IReadFile* _file) const
		{	
			const size_t begginingOfFile = _file->getPos();
            _file->seek(0ull);

			char magicNumberBuffer[sizeof(SContext::magicNumber)];
			_file->read(magicNumberBuffer, sizeof(SContext::magicNumber));
			_file->seek(begginingOfFile);

			return isImfMagic(magicNumberBuffer);
		}

		template<typename rgbaFormat>
		void readRgba(InputFile& file, std::array<Array2D<rgbaFormat>, 4>& pixelRgbaMapArray, int& width, int& height, E_FORMAT& format, const uint8_t& availableChannels)
		{
			Box2i dw = file.header().dataWindow();
			width = dw.max.x - dw.min.x + 1;
			height = dw.max.y - dw.min.y + 1;

			constexpr const char* rgbaSignatureAsText[] = {"R", "G", "B", "A"};
			for (auto& pixelChannelBuffer : pixelRgbaMapArray)
				pixelChannelBuffer.resizeErase(height, width);

			FrameBuffer frameBuffer;
			PixelType pixelType;

			if (format == EF_R16G16B16A16_SFLOAT)
				pixelType = PixelType::HALF;
			else if (format == EF_R32G32B32A32_SFLOAT)
				pixelType = PixelType::FLOAT;
			else if (format == EF_R32G32B32A32_UINT)
				pixelType = PixelType::UINT;

			for (uint8_t rgbaChannelIndex = 0; rgbaChannelIndex < availableChannels; ++rgbaChannelIndex)
				frameBuffer.insert
				(
					rgbaSignatureAsText[rgbaChannelIndex],                                                      // name
					Slice(pixelType,                                                                            // type
					(char*)(&(pixelRgbaMapArray[rgbaChannelIndex])[0][0] - dw.min.x - dw.min.y * width),        // base
					sizeof((pixelRgbaMapArray[rgbaChannelIndex])[0][0]) * 1,                                    // xStride
					sizeof((pixelRgbaMapArray[rgbaChannelIndex])[0][0]) * width,                                // yStride
					1, 1,                                                                                       // x/y sampling
					rgbaChannelIndex == 3 ? 1 : 0                                                               // default fillValue for channels that aren't present in file - 1 for alpha, otherwise 0
				));	

			file.setFrameBuffer(frameBuffer);
			file.readPixels(dw.min.y, dw.max.y);
		}

		bool specifyIrrlichtEndFormat(E_FORMAT& format, const InputFile& file, bool& doesItSupportAlphaChannel)
		{
			const IMF::Channel* RChannel = file.header().channels().findChannel("R");
			const IMF::Channel* GChannel = file.header().channels().findChannel("G");
			const IMF::Channel* BChannel = file.header().channels().findChannel("B");
			const IMF::Channel* AChannel = file.header().channels().findChannel("A");

			const IMF::Channel* XChannel = file.header().channels().findChannel("X");
			const IMF::Channel* YChannel = file.header().channels().findChannel("Y");
			const IMF::Channel* ZChannel = file.header().channels().findChannel("Z");

			if (XChannel && YChannel && ZChannel)
			{
				os::Printer::log("LOAD EXR: the file consist of not supported CIE XYZ channels", file.fileName(), ELL_ERROR);
				return false;
			}

			if (RChannel && GChannel && BChannel)
				os::Printer::log("LOAD EXR: loading RGB file", file.fileName(), ELL_INFORMATION);
			else if(RChannel && GChannel)
				os::Printer::log("LOAD EXR: loading RG file", file.fileName(), ELL_INFORMATION);
			else if(RChannel)
				os::Printer::log("LOAD EXR: loading R file", file.fileName(), ELL_INFORMATION);
			else 
				os::Printer::log("LOAD EXR: the file's channels are invalid to load", file.fileName(), ELL_ERROR);

			if (AChannel)
				doesItSupportAlphaChannel = true;
			else
				doesItSupportAlphaChannel = false;

			auto doesRGBFormatHaveTheSameFormatLikePassedToIt = [&](const PixelType ImfTypeToCompare)
			{
				return (RChannel->type == ImfTypeToCompare && GChannel->type == ImfTypeToCompare && BChannel->type == ImfTypeToCompare);
			};

			auto isAlphaChannelTheSameFormatLikePassedToItIfExsists = [&](const PixelType ImfTypeToCompare)
			{
				if (doesItSupportAlphaChannel)
				{
					if (AChannel->type == ImfTypeToCompare)
						return true;
					else
					{
						os::Printer::log("LOAD EXR: the file doesn't have the same alpha channel type in comparison of RGB channels", file.fileName(), ELL_ERROR);
						return false;
					}
				}
				else
					return true;
			};

			if (doesRGBFormatHaveTheSameFormatLikePassedToIt(PixelType::HALF))
			{
				format = EF_R16G16B16A16_SFLOAT;

				if (!isAlphaChannelTheSameFormatLikePassedToItIfExsists(PixelType::HALF))
					return false;
			}
			else if (doesRGBFormatHaveTheSameFormatLikePassedToIt(PixelType::FLOAT))
			{
				format = EF_R32G32B32A32_SFLOAT;

				if (!isAlphaChannelTheSameFormatLikePassedToItIfExsists(PixelType::FLOAT))
					return false;
			}
			else if (doesRGBFormatHaveTheSameFormatLikePassedToIt(PixelType::UINT))
			{
				format = EF_R32G32B32A32_UINT;

				if (!isAlphaChannelTheSameFormatLikePassedToItIfExsists(PixelType::UINT))
					return false;
			}
			else
				return false;

			return true;
		}

		bool readVersionField(io::IReadFile* _file, SContext& ctx)
		{
			RgbaInputFile file(_file->getFileName().c_str());
			auto& versionField = ctx.versionField;
			
			versionField.mainDataRegisterField = file.version();

			auto isTheBitActive = [&](uint16_t bitToCheck)
			{
				return (versionField.mainDataRegisterField & (1 << bitToCheck - 1));
			};

			versionField.fileFormatVersionNumber |= isTheBitActive(1) | isTheBitActive(2) | isTheBitActive(3) | isTheBitActive(4) | isTheBitActive(5) | isTheBitActive(6) | isTheBitActive(7) | isTheBitActive(8);

			if (!isTheBitActive(11) && !isTheBitActive(12))
			{
				versionField.Compoment.type = SContext::VersionField::Compoment::SINGLE_PART_FILE;

				if (isTheBitActive(9))
				{
					versionField.Compoment.singlePartFileCompomentSubTypes = SContext::VersionField::Compoment::TILES;
					os::Printer::log("LOAD EXR: the file consist of not supported tiles", file.fileName(), ELL_ERROR);
					return false;
				}
				else
					versionField.Compoment.singlePartFileCompomentSubTypes = SContext::VersionField::Compoment::SCAN_LINES;
			}
			else if (!isTheBitActive(9) && !isTheBitActive(11) && isTheBitActive(12))
			{
				versionField.Compoment.type = SContext::VersionField::Compoment::MULTI_PART_FILE;
				versionField.Compoment.singlePartFileCompomentSubTypes = SContext::VersionField::Compoment::SCAN_LINES_OR_TILES;
				os::Printer::log("LOAD EXR: the file is a not supported multi part file", file.fileName(), ELL_ERROR);
				return false;
			}

			if (!isTheBitActive(9) && isTheBitActive(11) && isTheBitActive(12))
			{
				versionField.doesItSupportDeepData = true;
				os::Printer::log("LOAD EXR: the file consist of not supported deep data", file.fileName(), ELL_ERROR);
				return false;
			}
			else
				versionField.doesItSupportDeepData = false;

			if (isTheBitActive(10))
				versionField.doesFileContainLongNames = true;
			else
				versionField.doesFileContainLongNames = false;

			return true;
		}

		bool readHeader(const char fileName[], SContext& ctx)
		{
			RgbaInputFile file(fileName);
			auto& attribs = ctx.attributes;
			auto& versionField = ctx.versionField;

			// There is an OpenEXR library implementation error associated with dynamic_cast<>
			// Since OpenEXR loader only cares about RGB and RGBA, there is no need for bellow at the moment

			/*

			attribs.channels = file.header().findTypedAttribute<Channel>("channels");
			attribs.compression = file.header().findTypedAttribute<Compression>("compression");
			attribs.dataWindow = file.header().findTypedAttribute<Box2i>("dataWindow");
			attribs.displayWindow = file.header().findTypedAttribute<Box2i>("displayWindow");
			attribs.lineOrder = file.header().findTypedAttribute<LineOrder>("lineOrder");
			attribs.pixelAspectRatio = file.header().findTypedAttribute<float>("pixelAspectRatio");
			attribs.screenWindowCenter = file.header().findTypedAttribute<V2f>("screenWindowCenter");
			attribs.screenWindowWidth = file.header().findTypedAttribute<float>("screenWindowWidth");

			if (versionField.Compoment.singlePartFileCompomentSubTypes == SContext::VersionField::Compoment::TILES)
				attribs.tiles = file.header().findTypedAttribute<TileDescription>("tiles");

			if (versionField.Compoment.type == SContext::VersionField::Compoment::MULTI_PART_FILE)
				attribs.view = file.header().findTypedAttribute<std::string>("view");

			if (versionField.Compoment.type == SContext::VersionField::Compoment::MULTI_PART_FILE || versionField.doesItSupportDeepData)
			{
				attribs.name = file.header().findTypedAttribute<std::string>("name");
				attribs.type = file.header().findTypedAttribute<std::string>("type");
				attribs.version = file.header().findTypedAttribute<int>("version");
				attribs.chunkCount = file.header().findTypedAttribute<int>("chunkCount");
				attribs.maxSamplesPerPixel = file.header().findTypedAttribute<int>("maxSamplesPerPixel");
			}

			*/

			return true;
		}
	}
}

#endif // _IRR_COMPILE_WITH_OPENEXR_LOADER_