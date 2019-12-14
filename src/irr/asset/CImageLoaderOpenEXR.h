// Copyright (C) 2009-2012 AnastaZIuk
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#pragma once

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_OPENEXR_LOADER_

#include "irr/asset/IAssetLoader.h"

#include "openexr/IlmBase/Imath/ImathBox.h"
#include "openexr/OpenEXR/IlmImf/ImfRgbaFile.h"
#include "openexr/OpenEXR/IlmImf/ImfStringAttribute.h"
#include "openexr/OpenEXR/IlmImf/ImfMatrixAttribute.h"
#include "openexr/OpenEXR/IlmImf/ImfArray.h"
#include <algorithm>
#include <iostream>
#include <string>

#include "openexr/OpenEXR/IlmImf/ImfNamespace.h"
namespace IMF = OPENEXR_IMF_NAMESPACE;
namespace IMATH = IMATH_NAMESPACE;

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

				// line offset table TODO

				// scan line blocks TODO
			};

			bool readVersionField(io::IReadFile* _file, SContext& ctx);
			bool readHeader(const char fileName[], SContext& ctx);

			void readRgba(const char fileName[], IMF::Array2D<IMF::Rgba>& pixels, int& width, int& height);
		};
	}
}

#endif // _IRR_COMPILE_WITH_OPENEXR_LOADER_
