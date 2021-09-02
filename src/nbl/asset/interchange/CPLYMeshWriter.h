// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_ASSET_PLY_MESH_WRITER_H_INCLUDED__
#define __NBL_ASSET_PLY_MESH_WRITER_H_INCLUDED__

#include <iomanip>

#include "nbl/asset/ICPUMeshBuffer.h"
#include "nbl/asset/interchange/IAssetWriter.h"

namespace nbl
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

        virtual bool writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override = nullptr) override;

    private:

        struct SContext
        {
            SAssetWriteContext writeContext;
            size_t fileOffset = 0;
        };

        void writeBinary(const asset::ICPUMeshBuffer* _mbuf, size_t _vtxCount, size_t _fcCount, asset::E_INDEX_TYPE _idxType, void* const _indices, bool _forceFaces, const bool _vaidToWrite[4], SContext& context) const;
        void writeText(const asset::ICPUMeshBuffer* _mbuf, size_t _vtxCount, size_t _fcCount, asset::E_INDEX_TYPE _idxType, void* const _indices, bool _forceFaces, const bool _vaidToWrite[4], SContext& context) const;

        void writeAttribBinary(SContext& context, asset::ICPUMeshBuffer* _mbuf, uint32_t _vaid, size_t _ix, size_t _cpa, bool flipAttribute = false) const;

        //! Creates new mesh buffer with the same attribute buffers mapped but with normalized types changed to corresponding true integer types.
        static core::smart_refctd_ptr<asset::ICPUMeshBuffer> createCopyMBuffNormalizedReplacedWithTrueInt(const asset::ICPUMeshBuffer* _mbuf);

        static std::string getTypeString(asset::E_FORMAT _t);

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

            system::future<size_t> future;
            context.writeContext.outputFile->write(future, str.c_str(), context.fileOffset, str.size());
            {
                const auto bytesWritten = future.get();
                context.fileOffset += bytesWritten;
            }
        }
};

} // end namespace
} // end namespace

#endif
