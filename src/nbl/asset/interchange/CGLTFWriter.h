// Copyright (C) 2020 AnastaZIuk
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in Nabla.h

#ifndef __NBL_ASSET_C_MESH_WRITER_GLTF__
#define __NBL_ASSET_C_MESH_WRITER_GLTF__

#include "BuildConfigOptions.h"

#ifdef _NBL_COMPILE_WITH_GLTF_WRITER_

#include "nbl/system/ISystem.h"
#include "nbl/system/IFile.h"
#include "nbl/asset/ICPUImageView.h"
#include "nbl/asset/interchange/IAssetWriter.h"

namespace nbl
{
namespace asset
{
//! glTF Writer capable of writing .gltf files
/*
			glTF bridges the gap between 3D content creation tools and modern 3D applications
			by providing an efficient, extensible, interoperable format for the transmission and loading of 3D content.
		*/

class CGLTFWriter final : public asset::IAssetWriter
{
protected:
    virtual ~CGLTFWriter() {}

public:
    CGLTFWriter() {}

    virtual const char** getAssociatedFileExtensions() const override
    {
        static const char* extensions[]{"gltf", nullptr};
        return extensions;
    }

    uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_MESH; }

    uint32_t getSupportedFlags() override { return asset::EWF_NONE; }

    uint32_t getForcedFlags() override { return asset::EWF_NONE; }

    bool writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override = nullptr) override;
};
}
}

#endif  // _NBL_COMPILE_WITH_GLTF_WRITER_
#endif  // __NBL_ASSET_C_MESH_WRITER_GLTF__
