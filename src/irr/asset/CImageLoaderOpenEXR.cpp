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

namespace irr
{
	namespace asset
	{
		asset::SAssetBundle CImageLoaderOpenEXR::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
		{
			if (!_file)
				return {};

			SContext ctx;
			const auto& fileName = _file->getFileName().c_str();

			if (!readMagicNumber())
				return {};

			readVersionField();
			readHeader(fileName, ctx);

			Array2D<Rgba> pixels;
			int width;
			int height;

			readRgba(fileName, pixels, width, height);

			// TODO put data into buffer, consider OpenEXR 2.0 supported saving layouts 
			return SAssetBundle{};
		}

		bool CImageLoaderOpenEXR::isALoadableFileFormat(io::IReadFile* _file) const
		{
			return true;
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

		bool CImageLoaderOpenEXR::readHeader(const char fileName[], SContext& ctx)
		{
			RgbaInputFile file(fileName);
			auto& attribs = ctx.attributes;

			attribs.channels = file.header().findTypedAttribute <ChannelsAttribute>("channels");
			attribs.compression = file.header().findTypedAttribute <CompressionAttribute>("compression");
			attribs.dataWindow = file.header().findTypedAttribute <Box2i>("dataWindow");
			attribs.displayWindow = file.header().findTypedAttribute <Box2i>("displayWindow");
			attribs.lineOrder = file.header().findTypedAttribute <LineOrderAttribute>("lineOrder");
			attribs.pixelAspectRatio = file.header().findTypedAttribute <float>("pixelAspectRatio");
			attribs.screenWindowCenter = file.header().findTypedAttribute <V2fAttribute>("screenWindowCenter");
			attribs.screenWindowWidth = file.header().findTypedAttribute <float>("screenWindowWidth");

			attribs.name = file.header().findTypedAttribute <string>("name");
			attribs.type = file.header().findTypedAttribute <string>("type");
			attribs.version = file.header().findTypedAttribute <int>("version");
			attribs.chunkCount = file.header().findTypedAttribute <int>("chunkCount");
			attribs.maxSamplesPerPixel = file.header().findTypedAttribute <int>("maxSamplesPerPixel");
			attribs.tiles = file.header().findTypedAttribute <TiledescAttribute>("tiles");
			attribs.view = file.header().findTypedAttribute <TextAttribute>("view");

			// TODO - perform validation if needed
			return true;
		}
	}
}