// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_ASSET_STL_MESH_WRITER_H_INCLUDED__
#define __NBL_ASSET_STL_MESH_WRITER_H_INCLUDED__

#include "nbl/asset/ICPUMesh.h"
#include "nbl/asset/interchange/IAssetWriter.h"

namespace nbl
{
namespace asset
{

//! class to write meshes, implementing a STL writer
class CSTLMeshWriter : public asset::IAssetWriter
{
    protected:
        virtual ~CSTLMeshWriter();

    public:
        CSTLMeshWriter();

        virtual const char** getAssociatedFileExtensions() const
        {
            static const char* ext[]{ "stl", nullptr };
            return ext;
        }

        virtual uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_MESH; }

        virtual uint32_t getSupportedFlags() override { return asset::EWF_BINARY; }

        virtual uint32_t getForcedFlags() { return 0u; }

        virtual bool writeAsset(io::IWriteFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override = nullptr) override;

    protected:
        // write binary format
        bool writeMeshBinary(io::IWriteFile* file, const asset::ICPUMesh* mesh, const SAssetWriteParams& _params);

        // write text format
        bool writeMeshASCII(io::IWriteFile* file, const asset::ICPUMesh* mesh, const SAssetWriteParams& _params);

        // create vector output with line end into string
        void getVectorAsStringLine(const core::vectorSIMDf& v, core::stringc& s) const;

        // write face information to file
        void writeFaceText(io::IWriteFile* file, const core::vectorSIMDf& v1,
            const core::vectorSIMDf& v2, const core::vectorSIMDf& v3, const SAssetWriteParams& _params);
};

} // end namespace
} // end namespace

#endif
