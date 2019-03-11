// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_STL_MESH_WRITER_H_INCLUDED__
#define __IRR_STL_MESH_WRITER_H_INCLUDED__

#include "IMeshWriter.h"


namespace irr
{
namespace scene
{
	class ISceneManager;

	//! class to write meshes, implementing a STL writer
	class CSTLMeshWriter : public IMeshWriter
	{
        protected:
            virtual ~CSTLMeshWriter();

        public:
            CSTLMeshWriter(scene::ISceneManager* smgr);

            //! Returns the type of the mesh writer
            virtual EMESH_WRITER_TYPE getType() const;

            //! writes a mesh
            virtual bool writeMesh(io::IWriteFile* file, scene::ICPUMesh* mesh, int32_t flags=EMWF_NONE);

        protected:
            // write binary format
            bool writeMeshBinary(io::IWriteFile* file, scene::ICPUMesh* mesh, int32_t flags);

            // write text format
            bool writeMeshASCII(io::IWriteFile* file, scene::ICPUMesh* mesh, int32_t flags);

            // create vector output with line end into string
            void getVectorAsStringLine(const core::vectorSIMDf& v, core::stringc& s) const;

            // write face information to file
            void writeFaceText(io::IWriteFile* file, const core::vectorSIMDf& v1,
                    const core::vectorSIMDf& v2, const core::vectorSIMDf& v3);

            scene::ISceneManager* SceneManager;
	};

} // end namespace
} // end namespace

#endif

