// Copyright (C) 2019-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#ifndef _NBL_ASSET_STL_MESH_WRITER_H_INCLUDED_
#define _NBL_ASSET_STL_MESH_WRITER_H_INCLUDED_


#include "nbl/asset/ICPUPolygonGeometry.h"
#include "nbl/asset/interchange/IGeometryWriter.h"


namespace nbl::asset
{

//! class to write meshes, implementing a STL writer
class CSTLMeshWriter : public IGeometryWriter
{
    protected:
        ~CSTLMeshWriter() override;

    public:
        CSTLMeshWriter();

        virtual const char** getAssociatedFileExtensions() const
        {
            static const char* ext[]{ "stl", nullptr };
            return ext;
        }

        virtual uint32_t getSupportedFlags() override { return asset::EWF_BINARY; }

        virtual uint32_t getForcedFlags() { return 0u; }

        virtual bool writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override = nullptr) override;

    private:
      using pos_t = hlsl::float32_t3;
      using normal_t = hlsl::float32_t3;

      struct SContext
      {
         SAssetWriteContext writeContext;
         size_t fileOffset;
      };

      // write binary format
      bool writeMeshBinary(const ICPUPolygonGeometry* geom, SContext* context);

      // write text format
      bool writeMeshASCII(const ICPUPolygonGeometry* geom, SContext* context);

      // create vector output with line end into string
      void getVectorAsStringLine(const pos_t& v, std::string& s) const;

      // write face information to file
      void writeFaceText(const pos_t& v1, const pos_t& v2, const pos_t& v3, SContext* context);
};

} // end namespace
#endif
