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

        void writeBinary(const ICPUPolygonGeometry* geom, const ICPUPolygonGeometry::SDataView* uvView, bool writeNormals, size_t vertexCount, const uint32_t* indices, size_t faceCount, SContext& context) const;
        void writeText(const ICPUPolygonGeometry* geom, const ICPUPolygonGeometry::SDataView* uvView, bool writeNormals, size_t vertexCount, const uint32_t* indices, size_t faceCount, SContext& context) const;

        template<typename T>
        void writeVectorAsText(SContext& context, const T* _vec, size_t _elementsToWrite, bool flipVectors = false) const
        {
			constexpr size_t xID = 0u;
            std::stringstream ss;
            ss << std::fixed;
			bool currentFlipOnVariable = false;
			for (size_t i = 0u; i < _elementsToWrite; ++i)
			{
				if (flipVectors && i == xID)
					currentFlipOnVariable = true;
				else
					currentFlipOnVariable = false;

					ss << std::setprecision(6) << _vec[i] * (currentFlipOnVariable ? -1 : 1) << " ";
			}
            auto str = ss.str();

            system::IFile::success_t succ;
            context.writeContext.outputFile->write(succ, str.c_str(), context.fileOffset, str.size());
            context.fileOffset += succ.getBytesProcessed();
        }
};

} // end namespace
#endif
