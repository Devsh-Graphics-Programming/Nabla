
// Copyright (C) 2009-2012 Gaz Davidson
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#pragma once

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_GLI_LOADER_

#include "irr/asset/IAssetLoader.h"
#include "gli/gli.hpp"

namespace irr
{
	namespace asset
	{
		//! Texture loader capable of loading in .ktx, .dds and .kmg file extensions
		class CGLILoader final : public asset::IAssetLoader
		{
		protected:
			virtual ~CGLILoader() {}

		public:
			CGLILoader() {}

			bool isALoadableFileFormat(io::IReadFile* _file) const override;

			const char** getAssociatedFileExtensions() const override
			{
				static const char* extensions[]{ "ktx", "dds", "kmg", nullptr };
				return extensions;
			}

			uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_IMAGE_VIEW; }

			asset::SAssetBundle loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;

		private:
			struct SContext
			{
				SContext(const size_t& sizeInBytes) : sourceCodeBuffer(core::make_smart_refctd_ptr<ICPUBuffer>(sizeInBytes)) {}
				io::IReadFile* file;
				core::smart_refctd_ptr<ICPUBuffer> sourceCodeBuffer;
			};

			inline static std::pair<E_FORMAT, ICPUImageView::SComponentMapping> getTranslatedGLIFormat(const gli::texture& texture, const gli::gl& glVersion);
		};
	}
}

#endif // _IRR_COMPILE_WITH_GLI_LOADER_