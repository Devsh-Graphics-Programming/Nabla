// Copyright (C) 2020 AnastaZIuk
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifdef _IRR_COMPILE_WITH_GLTF_WRITER_

#include "IrrCompileConfig.h"
#include "irr/asset/ICPUImageView.h"
#include "irr/asset/IAssetLoader.h"

namespace irr
{
	namespace asset
	{
		class CGLTFWriter final : public asset::IAssetWriter
		{
			protected:
				virtual ~CGLTFWriter() {}

			public:
				CGLTFWriter() {}

				virtual const char** getAssociatedFileExtensions() const override
				{
					static const char* extensions[]{ "gltf", nullptr };
					return extensions;
				}

				uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_MESH; }

				uint32_t getSupportedFlags() override { return asset::EWF_NONE; }

				uint32_t getForcedFlags() override { return asset::EWF_NONE; }

				bool writeAsset(io::IWriteFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override = nullptr) override;
		};
	}
}

#endif // _IRR_COMPILE_WITH_GLTF_WRITER_
