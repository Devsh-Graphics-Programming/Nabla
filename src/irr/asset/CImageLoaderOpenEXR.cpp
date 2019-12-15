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

namespace irr
{
	namespace asset
	{
		using namespace IMF;
		using namespace IMATH;

		asset::SAssetBundle CImageLoaderOpenEXR::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
		{
			if (!_file)
				return {};

			const auto& fileName = _file->getFileName().c_str();

			if (isALoadableFileFormat(_file))
				return {};

			SContext ctx;

			// readVersionField(_file, ctx);
			// readHeader(fileName, ctx);

			Array2D<Rgba> pixels;
			int width;
			int height;

			readRgba(fileName, pixels, width, height);

			constexpr uint32_t MAX_PITCH_ALIGNMENT = 8u;									   // OpenGL cannot transfer rows with arbitrary padding
			auto calcPitchInBlocks = [](uint32_t width, uint32_t blockByteSize) -> uint32_t	   // try with largest alignment first
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

			ICPUImage::SCreationParams params;
			params.format = EF_R16G16B16A16_SFLOAT;
			params.type = ICPUImage::ET_2D;;
			params.flags = static_cast<ICPUImage::E_CREATE_FLAGS>(0u);
			params.samples = ICPUImage::ESCF_1_BIT;
			params.extent.width = width;
			params.extent.height = height;
			params.extent.depth = 1u;
			params.mipLevels = 1u;
			params.arrayLayers = 1u;

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

			float* dataToSend = _IRR_NEW_ARRAY(float, image->getImageDataSizeInBytes());
			for (uint32_t i = 0; i < image->getImageDataSizeInBytes(); i += 4)
			{
				dataToSend[i] = pixels[i]->r;
				dataToSend[i + 1] = pixels[i]->g;
				dataToSend[i + 2] = pixels[i]->b;
				dataToSend[i + 3] = pixels[i]->a;
			}

			memcpy(texelBuffer->getPointer(), dataToSend, image->getImageDataSizeInBytes());
			image->setBufferAndRegions(std::move(texelBuffer), regions);

			_IRR_DELETE_ARRAY(dataToSend, image->getImageDataSizeInBytes());
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

		void CImageLoaderOpenEXR::readRgba(const char fileName[], Array2D<Rgba>& pixels, int& width, int& height)
		{
			RgbaInputFile file(fileName);
			Box2i dw = file.dataWindow();
			width = dw.max.x - dw.min.x + 1;
			height = dw.max.y - dw.min.y + 1;
			pixels.resizeErase(height, width);
			file.setFrameBuffer(&pixels[0][0] - dw.min.x - dw.min.y * width, 1, width);
			file.readPixels(dw.min.y, dw.max.y);
		}

		bool CImageLoaderOpenEXR::readVersionField(io::IReadFile* _file, CImageLoaderOpenEXR::SContext& ctx)
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

		bool CImageLoaderOpenEXR::readHeader(const char fileName[], CImageLoaderOpenEXR::SContext& ctx)
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