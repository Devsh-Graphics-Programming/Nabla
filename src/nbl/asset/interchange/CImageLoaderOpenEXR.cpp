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
#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>

#include "nbl/asset/IAssetManager.h"

#ifdef _NBL_COMPILE_WITH_OPENEXR_LOADER_

#include "nbl/asset/filters/CRegionBlockFunctorFilter.h"
#include "nbl/asset/metadata/COpenEXRMetadata.h"

#include "CImageLoaderOpenEXR.h"

#include "openexr/IlmBase/Imath/ImathBox.h"
#include "openexr/OpenEXR/IlmImf/ImfRgbaFile.h"
#include "openexr/OpenEXR/IlmImf/ImfInputFile.h"
#include "openexr/OpenEXR/IlmImf/ImfChannelList.h"
#include "openexr/OpenEXR/IlmImf/ImfChannelListAttribute.h"
#include "openexr/OpenEXR/IlmImf/ImfStringAttribute.h"
#include "openexr/OpenEXR/IlmImf/ImfMatrixAttribute.h"
#include "openexr/OpenEXR/IlmImf/ImfArray.h"

#include "openexr/OpenEXR/IlmImf/ImfNamespace.h"
namespace IMF = Imf;
namespace IMATH = Imath;

namespace nbl
{
	namespace asset
	{
		using namespace IMF;
		using namespace IMATH;

		namespace impl
		{
			class nblIStream : public IMF::IStream
			{
				public:
					nblIStream(system::IFile* _nblFile)
						: IMF::IStream(getFileName(_nblFile).c_str()), nblFile(_nblFile) {}
					virtual ~nblIStream() {}

					//------------------------------------------------------
					// Read from the stream:
					//
					// read(c,n) reads n bytes from the stream, and stores
					// them in array c.  If the stream contains less than n
					// bytes, or if an I/O error occurs, read(c,n) throws
					// an exception.  If read(c,n) reads the last byte from
					// the file it returns false, otherwise it returns true.
					//------------------------------------------------------

					virtual bool read(char c[/*n*/], int n) override
					{
						system::future<size_t> future;
						nblFile->read(future, c, fileOffset, n);
						const auto bytesRead = future.get();
						fileOffset += bytesRead;
						
						return true;
					}

					//--------------------------------------------------------
					// Get the current reading position, in bytes from the
					// beginning of the file.  If the next call to read() will
					// read the first byte in the file, tellg() returns 0.
					//--------------------------------------------------------

					virtual IMF::Int64 tellg() override
					{
						return static_cast<IMF::Int64>(fileOffset);
					}

					//-------------------------------------------
					// Set the current reading position.
					// After calling seekg(i), tellg() returns i.
					//-------------------------------------------

					virtual void seekg(IMF::Int64 pos) override
					{
						fileOffset = static_cast<decltype(fileOffset)>(pos);
					}

					//------------------------------------------------------
					// Clear error conditions after an operation has failed.
					//------------------------------------------------------

					virtual void clear() override
					{
						/*
							Probably we don't want to investigate in system::IFile
							and change the stream error state flags, leaving this 
							function empty
						*/
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

		using suffixOfChannelBundle = std::string;
		using channelName = std::string;	     									// sytnax if as follows
		using mapOfChannels = std::unordered_map<channelName, Channel>;				// suffix.channel, where channel are "R", "G", "B", "A"

		class SContext;
		bool readVersionField(IMF::IStream* nblIStream, SContext& ctx, const system::logger_opt_ptr);
		bool readHeader(IMF::IStream* nblIStream, SContext& ctx);
		template<typename rgbaFormat>
		void readRgba(InputFile& file, std::array<Array2D<rgbaFormat>, 4>& pixelRgbaMapArray, int& width, int& height, E_FORMAT& format, const suffixOfChannelBundle suffixOfChannels);
		E_FORMAT specifyIrrlichtEndFormat(const mapOfChannels& mapOfChannels, const suffixOfChannelBundle suffixName, const std::string fileName, const system::logger_opt_ptr logger);

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
				const Channel* channels = nullptr;
				const Compression* compression = nullptr;
				const Box2i* dataWindow = nullptr;
				const Box2i* displayWindow = nullptr;
				const LineOrder* lineOrder = nullptr;
				const float* pixelAspectRatio = nullptr;
				const V2f* screenWindowCenter = nullptr;
				const float* screenWindowWidth = nullptr;

				// These attributes are required in the header for all multi - part and /or deep data OpenEXR files
				const std::string* name = nullptr;
				const std::string* type = nullptr;
				const int* version = nullptr;
				const int* chunkCount = nullptr;

				// This attribute is required in the header for all files which contain deep data (deepscanline or deeptile)
				const int* maxSamplesPerPixel = nullptr;

				// This attribute is required in the header for all files which contain one or more tiles
				const TileDescription* tiles = nullptr;

				// This attribute can be used in the header for multi-part files
				const std::string* view = nullptr;

				// Others not required that can be used by metadata
				// - none at the moment

			} attributes;

			// core::smart_refctd_dynamic_array<uint32_t> offsetTable; 

			// scan line blocks
		};

		constexpr uint8_t availableChannels = 4;
		struct PerImageData
		{
			ICPUImage::SCreationParams params;
			std::array<Array2D<half>, availableChannels> halfPixelMapArray;
			std::array<Array2D<float>, availableChannels> fullFloatPixelMapArray;
			std::array<Array2D<uint32_t>, availableChannels> uint32_tPixelMapArray;
		};
		template<typename IlmType>
		struct ReadTexels
		{
				ReadTexels(ICPUImage* image, const std::array<Array2D<IlmType>, availableChannels>& _pixelMapArray) :
					data(reinterpret_cast<uint8_t*>(image->getBuffer()->getPointer())), pixelMapArray(_pixelMapArray)
				{
					using StreamFromEXR = CRegionBlockFunctorFilter<ReadTexels<IlmType>,false>;
					typename StreamFromEXR::state_type state(*this,image,image->getRegions().begin());
					StreamFromEXR::execute(&state);
				}

				inline void operator()(uint32_t ptrOffset, const core::vectorSIMDu32& texelCoord)
				{
					assert(texelCoord.w==0u && texelCoord.z==0u);

					uint8_t* texelPtr = data+ptrOffset;
					for (auto channelIndex=0; channelIndex<availableChannels; channelIndex++)
					{
						const auto& element = pixelMapArray[channelIndex][texelCoord.y][texelCoord.x];
						reinterpret_cast<typename std::decay<decltype(element)>::type*>(texelPtr)[channelIndex] = element;
					}
				}

			private:
				uint8_t* const data;
				const std::array<Array2D<IlmType>, availableChannels>& pixelMapArray;
		};

		auto getChannels(const InputFile& file)
		{
			std::unordered_map<suffixOfChannelBundle, mapOfChannels> irrChannels;		    // example: G, albedo.R, color.space.B
			{
				auto channels = file.header().channels();
				for (auto mapItr = channels.begin(); mapItr != channels.end(); ++mapItr)
				{
					std::string fetchedChannelName = mapItr.name();
					const bool isThereAnySuffix = fetchedChannelName.size() > 1;

					if (isThereAnySuffix)
					{
						const auto endPositionOfChannelName = fetchedChannelName.find_last_of(".");
						auto suffix = fetchedChannelName.substr(0, endPositionOfChannelName);
						auto channel = fetchedChannelName.substr(endPositionOfChannelName + 1);
						if (channel == "R" || channel == "G" || channel == "B" || channel == "A")
							(irrChannels[suffix])[channel] = mapItr.channel();
					}
					else
						(irrChannels[""])[fetchedChannelName] = mapItr.channel();
				}
			}

			return irrChannels;
		}

		auto doesTheChannelExist(const std::string channelName, const mapOfChannels& mapOfChannels)
		{
			auto foundPosition = mapOfChannels.find(channelName);
			if (foundPosition != mapOfChannels.end())
				return true;
			else
				return false;
		}

		SAssetBundle CImageLoaderOpenEXR::loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
		{
			if (!_file)
				return {};

			SContext ctx;

			IMF::IStream* nblIStream = _NBL_NEW(impl::nblIStream, _file); // TODO: THIS NEEDS TESTING
			InputFile file(*nblIStream);

			if (file.isComplete())
				static_cast<impl::nblIStream*>(nblIStream)->resetFileOffset();
			else
			{
				_NBL_DELETE(nblIStream);
				return {};
			}

			if (readVersionField(nblIStream, ctx, _params.logger))
				static_cast<impl::nblIStream*>(nblIStream)->resetFileOffset();
			else
			{
				_NBL_DELETE(nblIStream);
				return {};
			}

			if (readHeader(nblIStream, ctx))
				static_cast<impl::nblIStream*>(nblIStream)->resetFileOffset();
			else
			{
				_NBL_DELETE(nblIStream);
				return {};
			}

			_NBL_DELETE(nblIStream);

			core::vector<core::smart_refctd_ptr<ICPUImage>> images;
			const auto channelsData = getChannels(file);
			auto meta = core::make_smart_refctd_ptr<COpenEXRMetadata>(channelsData.size());
			{
				uint32_t metaOffset = 0u;
				for (const auto& data : channelsData)
				{
					const auto suffixOfChannels = data.first;
					const auto mapOfChannels = data.second;
					PerImageData perImageData;

					int width;
					int height;

					auto params = perImageData.params;
					params.format = specifyIrrlichtEndFormat(mapOfChannels, suffixOfChannels, file.fileName(), _params.logger);
					params.type = ICPUImage::ET_2D;;
					params.flags = static_cast<ICPUImage::E_CREATE_FLAGS>(0u);
					params.samples = ICPUImage::ESCF_1_BIT;
					params.extent.depth = 1u;
					params.mipLevels = 1u;
					params.arrayLayers = 1u;

					if (params.format == EF_UNKNOWN)
					{
						_params.logger.log("LOAD EXR: incorrect format specified for " + suffixOfChannels + " channels - skipping the file %s", system::ILogger::ELL_INFO, file.fileName());
						continue;
					}

					if (params.format == EF_R16G16B16A16_SFLOAT)
						readRgba(file, perImageData.halfPixelMapArray, width, height, params.format, suffixOfChannels);
					else if (params.format == EF_R32G32B32A32_SFLOAT)
						readRgba(file, perImageData.fullFloatPixelMapArray, width, height, params.format, suffixOfChannels);
					else if (params.format == EF_R32G32B32A32_UINT)
						readRgba(file, perImageData.uint32_tPixelMapArray, width, height, params.format, suffixOfChannels);

					params.extent.width = width;
					params.extent.height = height;

					auto image = ICPUImage::create(std::move(params));
					{ // create image and buffer that backs it
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

						image->setBufferAndRegions(std::move(texelBuffer), regions);
					}

					if (params.format == EF_R16G16B16A16_SFLOAT)
						ReadTexels(image.get(),perImageData.halfPixelMapArray);
					else if (params.format == EF_R32G32B32A32_SFLOAT)
						ReadTexels(image.get(), perImageData.fullFloatPixelMapArray);
					else if (params.format == EF_R32G32B32A32_UINT)
						ReadTexels(image.get(), perImageData.uint32_tPixelMapArray);

					meta->placeMeta(metaOffset++,image.get(),std::string(suffixOfChannels),IImageMetadata::ColorSemantic{ ECP_SRGB,EOTF_IDENTITY });

					images.push_back(std::move(image));
				}
			}	

			return SAssetBundle(std::move(meta),std::move(images));
		}

		bool CImageLoaderOpenEXR::isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger) const
		{	
			char magicNumberBuffer[sizeof(SContext::magicNumber)];
			system::future<size_t> future;
			_file->read(future, magicNumberBuffer, 0, sizeof(SContext::magicNumber));
			future.get();
			return isImfMagic(magicNumberBuffer);
		}

		template<typename rgbaFormat>
		void readRgba(InputFile& file, std::array<Array2D<rgbaFormat>, 4>& pixelRgbaMapArray, int& width, int& height, E_FORMAT& format, const suffixOfChannelBundle suffixOfChannels)
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
			{
				std::string name = suffixOfChannels.empty() ? rgbaSignatureAsText[rgbaChannelIndex] : suffixOfChannels + "." + rgbaSignatureAsText[rgbaChannelIndex];
				frameBuffer.insert
				(
					name.c_str(),																					// name
					Slice(pixelType,																				// type
					(char*)(&(pixelRgbaMapArray[rgbaChannelIndex])[0][0] - dw.min.x - dw.min.y * width),			// base
						sizeof((pixelRgbaMapArray[rgbaChannelIndex])[0][0]) * 1,                                    // xStride
						sizeof((pixelRgbaMapArray[rgbaChannelIndex])[0][0]) * width,                                // yStride
						1, 1,                                                                                       // x/y sampling
						rgbaChannelIndex == 3 ? 1 : 0                                                               // default fillValue for channels that aren't present in file - 1 for alpha, otherwise 0
					));
			}

			file.setFrameBuffer(frameBuffer);
			file.readPixels(dw.min.y, dw.max.y);
		}

		E_FORMAT specifyIrrlichtEndFormat(const mapOfChannels& mapOfChannels, const suffixOfChannelBundle suffixName, const std::string fileName, const system::logger_opt_ptr logger)
		{
			E_FORMAT retVal;

			const auto rChannel = doesTheChannelExist("R", mapOfChannels);
			const auto gChannel = doesTheChannelExist("G", mapOfChannels);
			const auto bChannel = doesTheChannelExist("B", mapOfChannels);
			const auto aChannel = doesTheChannelExist("A", mapOfChannels);
			
			if (rChannel && gChannel && bChannel && aChannel)
				logger.log("LOAD EXR: loading " + suffixName + " RGBA file %s", system::ILogger::ELL_INFO, fileName.c_str());
			else if (rChannel && gChannel && bChannel)
				logger.log("LOAD EXR: loading " + suffixName + " RGB file %s", system::ILogger::ELL_INFO, fileName.c_str());
			else if(rChannel && gChannel)
				logger.log("LOAD EXR: loading " + suffixName + " RG file %s", system::ILogger::ELL_INFO, fileName.c_str());
			else if(rChannel)
				logger.log("LOAD EXR: loading " + suffixName + " R file %s", system::ILogger::ELL_INFO, fileName.c_str());
			else 
				logger.log("LOAD EXR: the file's channels are invalid to load %s", system::ILogger::ELL_ERROR, fileName.c_str());

			auto doesMapOfChannelsFormatHaveTheSameFormatLikePassedToIt = [&](const PixelType ImfTypeToCompare)
			{
				for (auto& channel : mapOfChannels)
					if (channel.second.type != ImfTypeToCompare)
						return false;
				return true;
			};

			if (doesMapOfChannelsFormatHaveTheSameFormatLikePassedToIt(PixelType::HALF))
				retVal = EF_R16G16B16A16_SFLOAT;
			else if (doesMapOfChannelsFormatHaveTheSameFormatLikePassedToIt(PixelType::FLOAT))
				retVal = EF_R32G32B32A32_SFLOAT;
			else if (doesMapOfChannelsFormatHaveTheSameFormatLikePassedToIt(PixelType::UINT))
				retVal = EF_R32G32B32A32_UINT;
			else
				return EF_UNKNOWN;

			return retVal;
		}

		bool readVersionField(IMF::IStream* nblIStream, SContext& ctx, const system::logger_opt_ptr logger)
		{
			RgbaInputFile file(*nblIStream);

			if (!file.isComplete())
				return false;

			auto& versionField = ctx.versionField;
			
			versionField.mainDataRegisterField = file.version();

			auto isTheBitActive = [&](uint16_t bitToCheck)
			{
				return (versionField.mainDataRegisterField & (1 << (bitToCheck - 1)));
			};

			versionField.fileFormatVersionNumber |= isTheBitActive(1) | isTheBitActive(2) | isTheBitActive(3) | isTheBitActive(4) | isTheBitActive(5) | isTheBitActive(6) | isTheBitActive(7) | isTheBitActive(8);

			if (!isTheBitActive(11) && !isTheBitActive(12))
			{
				versionField.Compoment.type = SContext::VersionField::Compoment::SINGLE_PART_FILE;

				if (isTheBitActive(9))
				{
					versionField.Compoment.singlePartFileCompomentSubTypes = SContext::VersionField::Compoment::TILES;
					logger.log("LOAD EXR: the file consist of not supported tiles %s", system::ILogger::ELL_ERROR, file.fileName());
					return false;
				}
				else
					versionField.Compoment.singlePartFileCompomentSubTypes = SContext::VersionField::Compoment::SCAN_LINES;
			}
			else if (!isTheBitActive(9) && !isTheBitActive(11) && isTheBitActive(12))
			{
				versionField.Compoment.type = SContext::VersionField::Compoment::MULTI_PART_FILE;
				versionField.Compoment.singlePartFileCompomentSubTypes = SContext::VersionField::Compoment::SCAN_LINES_OR_TILES;
				logger.log("LOAD EXR: the file is a not supported multi part file %s", system::ILogger::ELL_ERROR, file.fileName());
				return false;
			}

			if (!isTheBitActive(9) && isTheBitActive(11) && isTheBitActive(12))
			{
				versionField.doesItSupportDeepData = true;
				logger.log("LOAD EXR: the file consist of not supported deep data%s", system::ILogger::ELL_ERROR, file.fileName());
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

		bool readHeader(IMF::IStream* nblIStream, SContext& ctx)
		{
			RgbaInputFile file(*nblIStream);

			if (!file.isComplete())
				return false;

			auto& attribs = ctx.attributes;
			auto& versionField = ctx.versionField;

			/*

			// There is an OpenEXR library implementation error associated with dynamic_cast<> probably
			// Since OpenEXR loader only cares about RGB and RGBA, there is no need for bellow at the moment

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

#endif // _NBL_COMPILE_WITH_OPENEXR_LOADER_
