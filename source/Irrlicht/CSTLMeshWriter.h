// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_STL_MESH_WRITER_H_INCLUDED__
#define __IRR_STL_MESH_WRITER_H_INCLUDED__

#include "IAssetWriter.h"


namespace irr
{
namespace scene
{
	class ISceneManager;

	//! class to write meshes, implementing a STL writer
	class CSTLMeshWriter : public asset::IAssetWriter
	{
        protected:
            virtual ~CSTLMeshWriter();

        public:
            CSTLMeshWriter(scene::ISceneManager* smgr);

            virtual const char** getAssociatedFileExtensions() const
            {
                static const char* ext[]{ "ply", nullptr };
                return ext;
            }

            virtual uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_MESH; }

            virtual uint32_t getSupportedFlags() override { return asset::EWF_BINARY; }

            virtual uint32_t getForcedFlags() { return 0u; }

            virtual bool writeAsset(io::IWriteFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override = nullptr) override;

        protected:
            // write binary format
            bool writeMeshBinary(io::IWriteFile* file, const scene::ICPUMesh* mesh);

            // write text format
            bool writeMeshASCII(io::IWriteFile* file, const scene::ICPUMesh* mesh);

            // create vector output with line end into string
            void getVectorAsStringLine(const core::vector3df& v,
                    core::stringc& s) const;

            // write face information to file
            void writeFaceText(io::IWriteFile* file, const core::vector3df& v1,
                    const core::vector3df& v2, const core::vector3df& v3);

            scene::ISceneManager* SceneManager;
	};

} // end namespace
} // end namespace

#endif

