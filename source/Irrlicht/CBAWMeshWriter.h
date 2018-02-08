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
			core::array<core::BlobHeaderV0> headers;
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

		//! Generates header of blobs from mesh object and pushes them to `SContext::headers`.
		/** After calling this method headers are NOT ready yet. Hashes (and also size in case of texture path blob) are calculated while writing blob data.
		@param _mesh Pointer to the mesh object.
		@return Amount of generated headers.*/
		uint32_t genHeaders(ICPUMesh* _mesh, SContext& _ctx);

		//! Pushes new offset value to `SContext::offsets` array.
		/** @param _blobSize Byte-distance from previous blob's first byte (i.e. size of previous blob).
		*/
		void calcAndPushNextOffset(uint32_t _blobSize, SContext& _ctx) const;

		//! Pushes corrupted offset so that, while loading resulting .baw file, it will be easy to find out something went wrong.
		void pushCorruptedOffset(SContext& _ctx) const { _ctx.offsets.push_back(0xffffffff); }

		//! Tries to write given data to file. If not possible (i.e. _data is NULL) - pushes "corrupted offset" and does not call .finalize() on blob-header.
		void tryWrite(const void* _data, io::IWriteFile* _file, SContext& _ctx, size_t _size, uint32_t _headerIdx) const;

	private:
		io::IFileSystem* m_fileSystem;

		static const char * const BAW_FILE_HEADER;
	};

}
} // end of ns irr:scene

#endif