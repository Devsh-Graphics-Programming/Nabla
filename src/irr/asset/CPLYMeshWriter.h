// Copyright (C) 2009-2012 Gaz Davidson
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_PLY_MESH_WRITER_H_INCLUDED__
#define __IRR_PLY_MESH_WRITER_H_INCLUDED__

#include "irr/asset/IAssetWriter.h"
#include "irr/video/IGPUMeshBuffer.h"
#include <iomanip>

namespace irr
{

namespace asset
{
	//! class to write PLY mesh files
	class CPLYMeshWriter : public asset::IAssetWriter
	{
	public:

		CPLYMeshWriter();

        virtual const char** getAssociatedFileExtensions() const
        {
            static const char* ext[]{ "ply", nullptr };
            return ext;
        }

        virtual uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_MESH; }

        virtual uint32_t getSupportedFlags() override { return asset::EWF_BINARY; }

        virtual uint32_t getForcedFlags() { return 0u; }

        virtual bool writeAsset(io::IWriteFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override = nullptr) override;

    private:
        void writeBinary(io::IWriteFile* _file, asset::ICPUMeshBuffer* _mbuf, size_t _vtxCount, size_t _fcCount, asset::E_INDEX_TYPE _idxType, void* const _indices, bool _forceFaces, const bool _vaidToWrite[4]) const;
        void writeText(io::IWriteFile* _file, asset::ICPUMeshBuffer* _mbuf, size_t _vtxCount, size_t _fcCount, asset::E_INDEX_TYPE _idxType, void* const _indices, bool _forceFaces, const bool _vaidToWrite[4]) const;

        void writeAttribBinary(io::IWriteFile* _file, asset::ICPUMeshBuffer* _mbuf, asset::E_VERTEX_ATTRIBUTE_ID _vaid, size_t _ix, size_t _cpa) const;

        //! Creates new mesh buffer with the same attribute buffers mapped but with normalized types changed to corresponding true integer types.
        static asset::ICPUMeshBuffer* createCopyMBuffNormalizedReplacedWithTrueInt(const asset::ICPUMeshBuffer* _mbuf);

        static std::string getTypeString(asset::E_FORMAT _t);

        template<typename T>
        void writeVectorAsText(io::IWriteFile* _file, const T* _vec, size_t _elementsToWrite) const
        {
            std::stringstream ss;
            ss << std::fixed;
            for (size_t i = 0u; i < _elementsToWrite; ++i)
                ss << std::setprecision(6) << _vec[i] << " ";
            auto str = ss.str();
            _file->write(str.c_str(), str.size());
        }
	};

} // end namespace
} // end namespace

#endif
