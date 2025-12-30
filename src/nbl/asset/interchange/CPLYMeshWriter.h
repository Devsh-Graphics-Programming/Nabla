// Copyright (C) 2019-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#ifndef _NBL_ASSET_PLY_MESH_WRITER_H_INCLUDED_
#define _NBL_ASSET_PLY_MESH_WRITER_H_INCLUDED_


#include "nbl/asset/ICPUPolygonGeometry.h"
#include "nbl/asset/interchange/IGeometryWriter.h"

#include <iomanip>


namespace nbl::asset
{

//! class to write PLY mesh files
class CPLYMeshWriter : public IGeometryWriter
{
	public:
		CPLYMeshWriter();

        virtual const char** getAssociatedFileExtensions() const
        {
            static const char* ext[]{ "ply", nullptr };
            return ext;
        }

        virtual uint32_t getSupportedFlags() override { return asset::EWF_BINARY; }

        virtual uint32_t getForcedFlags() { return 0u; }

        virtual bool writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override = nullptr) override;

    private:

        struct SContext
        {
            SAssetWriteContext writeContext;
            size_t fileOffset = 0;
        };

        void writeBinary(const ICPUPolygonGeometry* geom, const bool attrsToWrite[4], bool _forceFaces, SContext& context) const;
        void writeText(const ICPUPolygonGeometry* geom, const bool attrsToWrite[4], bool _forceFaces, SContext& context) const;

        void writeAttribBinary(SContext& context, ICPUPolygonGeometry* geom, uint32_t _vaid, size_t _ix, size_t _cpa, bool flipAttribute = false) const;

        //! Creates new geometry with the same attribute buffers mapped but with normalized types changed to corresponding true integer types.
        static core::smart_refctd_ptr<ICPUPolygonGeometry> createCopyNormalizedReplacedWithTrueInt(const ICPUPolygonGeometry* geom);

        static std::string getTypeString(asset::E_FORMAT _t);

        inline void writeVertexAsText(SContext& context, const hlsl::float32_t3& pos, const hlsl::float32_t3* normal) const
        {
           std::stringstream ss;
           ss << std::fixed << std::setprecision(6) << pos.x << " " << pos.y << " " << pos.z;

           if (normal)
           {
              ss << " " << normal->x << " " << normal->y << " " << normal->z << "\n";
           }
           else
              ss << "\n";

           const auto& str = ss.str();

           system::IFile::success_t success;
           context.writeContext.outputFile->write(success, str.data(), context.fileOffset, str.size());
           context.fileOffset += success.getBytesProcessed();
        }
};

} // end namespace
#endif
