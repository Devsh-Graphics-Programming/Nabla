// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __IRR_BAW_MESH_WRITER_H_INCLUDED__
#define __IRR_BAW_MESH_WRITER_H_INCLUDED__

#include "IMeshWriter.h"
#include "IMesh.h"
#include "CBawFile.h"
#include "irrArray.h"

namespace irr {

namespace io { class IFileSystem; }

namespace scene
{
	class ISceneManager;

	class CBAWMeshWriter : public IMeshWriter
	{
	private:
		struct SContext
		{
			core::array<core::BlobHeader> headers;
			core::array<uint32_t> offsets;
		};

	protected:
		~CBAWMeshWriter() {}

	public:
		explicit CBAWMeshWriter(io::IFileSystem*);

		//! @copydoc irr::scene::IMeshWriter::getType()
		EMESH_WRITER_TYPE getType() const { return EMWT_BAW; }

		//! @copydoc irr::scene::IMeshWriter::writeMesh()
		bool writeMesh(io::IWriteFile* file, scene::ICPUMesh* mesh, int32_t flags = EMWF_NONE);
		
	private:
		//! Takes object and exports (writes to file) its data as another blob.
		/** @param _obj Pointer to object which is to be exported.
		@param _headersIdx Corresponding index of headers array.
		@param _file Target file.*/
		template<typename T>
		void exportAsBlob(T* _obj, uint32_t _headerIdx, io::IWriteFile* _file, SContext& _ctx);

		//! Generates header of blobs from mesh object and pushes them to `m_headers`.
		/** After calling this method headers are NOT ready yet. Hashes (and also size in case of texture path blob) are calculated while writing blob data.
		@param _mesh Pointer to the mesh object.
		@return Amount of generated headers.*/
		uint32_t genHeaders(ICPUMesh * _mesh, SContext& _ctx);

		//! Pushes new offset value to `m_offsets` array.
		/** @param _blobSize Byte-distance from previous blob's first byte (i.e. size of previous blob).
		*/
		void calcAndPushNextOffset(uint32_t _blobSize, SContext& _ctx);

	private:
		io::IFileSystem* m_fileSystem;

		static const char * const BAW_FILE_HEADER;
	};

}
} // end of ns irr:scene

#endif