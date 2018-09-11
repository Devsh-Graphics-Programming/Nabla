// Copyright (C) 2009-2012 Gaz Davidson
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_PLY_MESH_WRITER_H_INCLUDED__
#define __IRR_PLY_MESH_WRITER_H_INCLUDED__

#include "IMeshWriter.h"
#include <iomanip>

namespace irr
{

namespace scene
{
    class ICPUMeshBuffer;

	//! class to write PLY mesh files
	class CPLYMeshWriter : public IMeshWriter
	{
	public:

		CPLYMeshWriter();

		//! Returns the type of the mesh writer
		virtual EMESH_WRITER_TYPE getType() const;

		//! writes a mesh
		virtual bool writeMesh(io::IWriteFile* file, scene::ICPUMesh* mesh, int32_t flags=EMWF_NONE);

    private:
        void writeBinary(io::IWriteFile* _file, ICPUMeshBuffer* _mbuf, size_t _vtxCount, size_t _fcCount, E_INDEX_TYPE _idxType, void* const _indices, bool _forceFaces, const bool _vaidToWrite[4]) const;
        void writeText(io::IWriteFile* _file, ICPUMeshBuffer* _mbuf, size_t _vtxCount, size_t _fcCount, E_INDEX_TYPE _idxType, void* const _indices, bool _forceFaces, const bool _vaidToWrite[4]) const;

        void writeAttribBinary(io::IWriteFile* _file, ICPUMeshBuffer* _mbuf, E_VERTEX_ATTRIBUTE_ID _vaid, size_t _ix, size_t _cpa) const;

        //! Creates new mesh buffer with the same attribute buffers mapped but with normalized types changed to corresponding true integer types.
        static ICPUMeshBuffer* createCopyMBuffNormalizedReplacedWithTrueInt(const ICPUMeshBuffer* _mbuf);

        static std::string getTypeString(E_COMPONENT_TYPE _t);

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
