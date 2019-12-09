// Copyright (C) 2009-2012 AnastaZIuk
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#pragma once

#include "irr/asset/IAssetLoader.h"

#include "openexr/IlmBase/Imath/ImathBox.h"
#include "openexr/OpenEXR/IlmImf/ImfRgbaFile.h"
#include "openexr/OpenEXR/IlmImf/ImfStringAttribute.h"
#include "openexr/OpenEXR/IlmImf/ImfMatrixAttribute.h"
#include "openexr/OpenEXR/IlmImf/ImfArray.h"
#include <algorithm>

#include "openexr/OpenEXR/IlmImf/ImfNamespace.h"
namespace IMF = OPENEXR_IMF_NAMESPACE;
namespace IMATH = IMATH_NAMESPACE;

using namespace IMF;
using namespace IMATH;

namespace irr
{
	namespace asset
	{
		//! OpenEXR loader capable of loading .exr files
		class CImageLoaderOpenEXR final : public asset::IAssetLoader
		{
		protected:
			~CImageLoaderOpenEXR(){}

		public:
			CImageLoaderOpenEXR(){}

			bool isALoadableFileFormat(io::IReadFile* _file) const override;

			const char** getAssociatedFileExtensions() const override
			{
				static const char* extensions[]{ "exr", nullptr };
				return extensions;
			}

			uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_IMAGE; }

			asset::SAssetBundle loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;

		private:

			struct SContext
			{
				constexpr static std::array<char, 4> magicNumber = { 0x76, 0x2f, 0x31, 0x01 };

				struct VersionField
				{
					int registerBitField = {};
					// TODO - pull out the bits and take them into account
				} versionField;

				struct Attributes
				{
					// The header of every OpenEXR file must contain at least the following attributes
					//according to https://www.openexr.com/documentation/openexrfilelayout.pdf (page 8)

					const ChannelsAttribute* channels = nullptr;
					const CompressionAttribute* compression = nullptr;
					const Box2i* dataWindow = nullptr;
					const Box2i* displayWindow = nullptr;
					const LineOrderAttribute* lineOrder = nullptr;
					const float* pixelAspectRatio = nullptr;
					const V2fAttribute* screenWindowCenter = nullptr;
					const float* screenWindowWidth = nullptr;

					// These attributes are required in the header for all multi - part and /or deep data OpenEXR files

					const string* name = nullptr;
					const string* type = nullptr;
					const int* version = nullptr;
					const int* chunkCount = nullptr;

					// This attribute is required in the header for all files which contain deep data (deepscanline or deeptile)

					const int* maxSamplesPerPixel = nullptr;

					// This attribute is required in the header for all files which contain one or more tiles

					const TiledescAttribute* tiles = nullptr;

					// This attribute can be used in the header for multi-part files
					const TextAttribute* view = nullptr;

					// Others not required that can be used by metadata
					// - none at the moment

				} attributes;

				// line offset table TODO

				// scan line blocks TODO
			};

			bool readMagicNumber() { return true; } // TODO
			bool readVersionField() { return true; } // TODO
			bool readHeader(const char fileName[], SContext& ctx);

			void readRgba(const char fileName[], Array2D<Rgba>& pixels, int& width, int& height);
		};
	}
}
