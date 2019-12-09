// Copyright (C) 2009-2012 AnastaZIuk
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#pragma once

#include "irr/asset/IAssetWriter.h"

namespace irr
{
	namespace asset
	{
		//! OpenEXR writer capable of saving .exr files
		class CImageWriterOpenEXR final : public asset::IAssetWriter
		{
		protected:
			~CImageWriterOpenEXR(){}

		public:
			CImageWriterOpenEXR(){}

			const char** getAssociatedFileExtensions() const override
			{
				static const char* extensions[]{ "exr", nullptr };
				return extensions;
			}

			uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_IMAGE; }

			uint32_t getSupportedFlags() override { return asset::EWF_BINARY; }

			uint32_t getForcedFlags() { return asset::EWF_BINARY; }

			bool writeAsset(io::IWriteFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override = nullptr) override;

		private:
		};
	}
}
