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
		void readRgba(InputFile& file, std::array<Array2D<rgbaFormat>, 4>& pixelRgbaMapArray, int& width, int& height, E_FORMAT& format);
		void specifyIrrlichtEndFormat(E_FORMAT& format, const InputFile& file);

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

			// scan line blocks TODO
		};

		template<typename IlmF>
		void assignToReinterpretedValue(E_FORMAT currentIrrlichtBaseFormatToCompareWith, void* begginingOfImageDataBuffer, const uint64_t& endShiftToSpecifyADataPos, const IlmF& ilmPixelValueToAssignTo)
		{
			switch (currentIrrlichtBaseFormatToCompareWith)
			{
				case EF_R16G16B16A16_SFLOAT:
				{
					*(reinterpret_cast<half*>(begginingOfImageDataBuffer) + endShiftToSpecifyADataPos) = ilmPixelValueToAssignTo;
					break;
				}
				case EF_R32G32B32A32_SFLOAT:
				{
					*(reinterpret_cast<float*>(begginingOfImageDataBuffer) + endShiftToSpecifyADataPos) = ilmPixelValueToAssignTo;
					break;
				}
				case EF_R32G32B32A32_UINT:
				{
					*(reinterpret_cast<uint32_t*>(begginingOfImageDataBuffer) + endShiftToSpecifyADataPos) = ilmPixelValueToAssignTo;
					break;
				}
			}
		};

		asset::SAssetBundle CImageLoaderOpenEXR::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
		{
			if (!_file)
				return {};

			const auto& fileName = _file->getFileName().c_str();

			if (isALoadableFileFormat(_file))
				return {};

			SContext ctx;
			InputFile file = fileName;

			// readVersionField(_file, ctx);
			// readHeader(fileName, ctx);

			std::array<Array2D<half>, 4> rgbaHalfPixelMapArray;
			std::array<Array2D<float>, 4> rgbaFullFloatPixelMapArray;
			std::array<Array2D<uint32_t>, 4> rgbaUint32_tPixelMapArray;

			int width;
			int height;

			ICPUImage::SCreationParams params;
			params.type = ICPUImage::ET_2D;;
			params.flags = static_cast<ICPUImage::E_CREATE_FLAGS>(0u);
			params.samples = ICPUImage::ESCF_1_BIT;
			params.extent.depth = 1u;
			params.mipLevels = 1u;
			params.arrayLayers = 1u;

			specifyIrrlichtEndFormat(params.format, file);

			switch (params.format)
			{
				case EF_R16G16B16A16_SFLOAT:
				{
					readRgba(file, rgbaHalfPixelMapArray, width, height, params.format);
					break;
				}
				case EF_R32G32B32A32_SFLOAT:
				{
					readRgba(file, rgbaFullFloatPixelMapArray, width, height, params.format);
					break;
				}
				case EF_R32G32B32A32_UINT:
				{
					readRgba(file, rgbaUint32_tPixelMapArray, width, height, params.format);
					break;
				}
			}

			params.extent.width = width;
			params.extent.height = height;

			static const uint32_t MAX_PITCH_ALIGNMENT = 8u;											// OpenGL cannot transfer rows with arbitrary padding
			auto calcPitchInBlocks = [](uint32_t width, uint32_t blockByteSize) -> uint32_t			// try with largest alignment first
			{
				auto rowByteSize = width * blockByteSize;
				for (uint32_t _alignment = MAX_PITCH_ALIGNMENT; _alignment > 1u; _alignment >>= 1u)
				{
					auto paddedSize = core::alignUp(rowByteSize, _alignment);
					if (paddedSize % blockByteSize)
						continue;
					return paddedSize / blockByteSize;
				}
				return width;
			};

			auto& image = ICPUImage::create(std::move(params));

			auto texelBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(image->getImageDataSizeInBytes());
			auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1u);
			ICPUImage::SBufferCopy& region = regions->front();
			//region.imageSubresource.aspectMask = ...; // waits for Vulkan
			region.imageSubresource.mipLevel = 0u;
			region.imageSubresource.baseArrayLayer = 0u;
			region.imageSubresource.layerCount = 1u;
			region.bufferOffset = 0u;
			region.bufferRowLength = calcPitchInBlocks(width, getTexelOrBlockBytesize(image->getCreationParameters().format));
			region.bufferImageHeight = 0u;
			region.imageOffset = { 0u, 0u, 0u };
			region.imageExtent = image->getCreationParameters().extent;

			void* fetchedData = texelBuffer->getPointer();

			for (uint64_t xPos = 0; xPos < width; ++xPos)
				for (uint64_t yPos = 0; yPos < height; ++yPos)
					for (uint8_t channelIndex = 0; channelIndex < 4; ++channelIndex)
					{ 
						const uint64_t ptrStyleEndShiftToImageDataPixel = (yPos * width) + xPos;

						switch (params.format)
						{
							case EF_R16G16B16A16_SFLOAT:
							{																	
								assignToReinterpretedValue(params.format, fetchedData, ptrStyleEndShiftToImageDataPixel + channelIndex, (rgbaHalfPixelMapArray[channelIndex])[xPos][yPos]);				
								break;
							}
							case EF_R32G32B32A32_SFLOAT:
							{
								assignToReinterpretedValue(params.format, fetchedData, ptrStyleEndShiftToImageDataPixel + channelIndex, (rgbaFullFloatPixelMapArray[channelIndex])[xPos][yPos]);
								break;
							}
							case EF_R32G32B32A32_UINT:
							{
								assignToReinterpretedValue(params.format, fetchedData, ptrStyleEndShiftToImageDataPixel + channelIndex, (rgbaUint32_tPixelMapArray[channelIndex])[xPos][yPos]);
								break;
							}
						}
					}

			image->setBufferAndRegions(std::move(texelBuffer), regions);

			return SAssetBundle{image};
		}

		bool CImageLoaderOpenEXR::isALoadableFileFormat(io::IReadFile* _file) const
		{
			const size_t begginingOfFile = _file->getPos();

			unsigned char magicNumberBuffer[sizeof(SContext::magicNumber)];
			_file->read(magicNumberBuffer, sizeof(SContext::magicNumber));
			_file->seek(begginingOfFile);

			auto deserializeToReadMagicValue = [&](unsigned char* buffer)
			{
				uint32_t value = 0ul;
				value |= buffer[0] << 24 | buffer[1] << 16 | buffer[2] << 8 | buffer[3];
				return value;
			};

			auto magicNumberToCompare = deserializeToReadMagicValue(magicNumberBuffer);
			if (magicNumberToCompare == SContext::magicNumber)
				return true;
			else
				return false;
		}

		template<typename rgbaFormat>
		void readRgba(InputFile& file, std::array<Array2D<rgbaFormat>, 4>& pixelRgbaMapArray, int& width, int& height, E_FORMAT& format)
		{
			Box2i dw = file.header().dataWindow();
			width = dw.max.x - dw.min.x + 1;
			height = dw.max.y - dw.min.y + 1;

			const std::string rgbaSignatureAsText = "RGBA";
			for (auto& pixelChannelBuffer : pixelRgbaMapArray)
				pixelChannelBuffer.resizeErase(height, width);

			FrameBuffer frameBuffer;
			PixelType pixelType;

			if (format == EF_R16G16B16A16_SFLOAT)
				pixelType = PixelType::HALF;
			else if (format == EF_R32G32B32A32_SFLOAT)
				pixelType = PixelType::FLOAT;
			else if (EF_R32G32B32A32_UINT)
				pixelType = PixelType::UINT;

			for (uint8_t rgbaChannelIndex = 0; rgbaChannelIndex < 4; ++rgbaChannelIndex)
				frameBuffer.insert
				(
					reinterpret_cast<const char*>(rgbaSignatureAsText[rgbaChannelIndex]),                       // name
					Slice(pixelType,                                                                            // type
					(char*)(&(pixelRgbaMapArray[rgbaChannelIndex])[0][0] - dw.min.x - dw.min.y * width),        // base
					sizeof((pixelRgbaMapArray[rgbaChannelIndex])[0][0]) * 1,                                    // xStride
					sizeof((pixelRgbaMapArray[rgbaChannelIndex])[0][0]) * width,                                // yStride
					1, 1,                                                                                       // x/y sampling
					0                                                                                           // fillValue
				));	

			file.setFrameBuffer(frameBuffer);
			file.readPixels(dw.min.y, dw.max.y);
		}

		void specifyIrrlichtEndFormat(E_FORMAT& format, const InputFile& file)
		{
			const IMF::Channel* RChannel = file.header().channels().findChannel("R");
			const IMF::Channel* GChannel = file.header().channels().findChannel("G");
			const IMF::Channel* BChannel = file.header().channels().findChannel("B");
			const IMF::Channel* AChannel = file.header().channels().findChannel("A");

			auto doesRGBAFormatHaveTheSameFormatLikePassedToIt = [&](const PixelType ImfTypeToCompare)
			{
				return (RChannel->type == ImfTypeToCompare && GChannel->type == ImfTypeToCompare && BChannel->type == ImfTypeToCompare && AChannel->type == ImfTypeToCompare);
			};

			if (doesRGBAFormatHaveTheSameFormatLikePassedToIt(PixelType::HALF))
				format = EF_R16G16B16A16_SFLOAT;
			else if (doesRGBAFormatHaveTheSameFormatLikePassedToIt(PixelType::FLOAT))
				format = EF_R32G32B32A32_SFLOAT;
			else if (doesRGBAFormatHaveTheSameFormatLikePassedToIt(PixelType::UINT))
				format = EF_R32G32B32A32_UINT;
			else
				assert(0);
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
					versionField.Compoment.singlePartFileCompomentSubTypes = SContext::VersionField::Compoment::TILES;
				else
					versionField.Compoment.singlePartFileCompomentSubTypes = SContext::VersionField::Compoment::SCAN_LINES;
			}
			else if (!isTheBitActive(9) && !isTheBitActive(11) && isTheBitActive(12))
			{
				versionField.Compoment.type = SContext::VersionField::Compoment::MULTI_PART_FILE;
				versionField.Compoment.singlePartFileCompomentSubTypes = SContext::VersionField::Compoment::SCAN_LINES_OR_TILES;
			}

			if (!isTheBitActive(9) && isTheBitActive(11) && isTheBitActive(12))
				versionField.doesItSupportDeepData = true;
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

			/*
			
			There is an OpenEXR library implementation error associated with dynamic_cast<>

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

#endif // #ifdef _IRR_COMPILE_WITH_OPENEXR_LOADER_