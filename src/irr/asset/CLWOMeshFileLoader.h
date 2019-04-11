// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_LWO_MESH_FILE_LOADER_H_INCLUDED__
#define __C_LWO_MESH_FILE_LOADER_H_INCLUDED__

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_LWO_LOADER_
#include "IMeshLoader.h"



namespace irr
{
namespace io
{
	class IReadFile;
	class IFileSystem;
} // end namespace io
namespace scene
{

	struct SMesh;
	class ISceneManager;

//! Meshloader capable of loading Lightwave 3D meshes.
class CLWOMeshFileLoader : public IMeshLoader
{
protected:
	//! destructor
	virtual ~CLWOMeshFileLoader();

public:
	//! Constructor
	CLWOMeshFileLoader(scene::ISceneManager* smgr, io::IFileSystem* fs);

	//! returns true if the file maybe is able to be loaded by this class
	//! based on the file extension (e.g. ".bsp")
	virtual bool isALoadableFileExtension(const io::path& filename) const;

	//! creates/loads an animated mesh from the file.
	//! \return Pointer to the created mesh. Returns 0 if loading failed.
	//! If you no longer need the mesh, you should call IAnimatedMesh::drop().
	//! See IUnknown::drop() for more information.
	virtual IAnimatedMesh* createMesh(io::IReadFile* file);

private:

	struct tLWOMaterial;

	bool readFileHeader();
	bool readChunks();
	void readObj1(uint32_t size);
	void readTagMapping(uint32_t size);
	void readVertexMapping(uint32_t size);
	void readDiscVertexMapping (uint32_t size);
	void readObj2(uint32_t size);
	void readMat(uint32_t size);
	uint32_t readString(core::stringc& name, uint32_t size=0);
	uint32_t readVec(core::vector3df& vec);
	uint32_t readVX(uint32_t& num);
	uint32_t readColor(video::SColor& color);
	video::ITexture* loadTexture(const core::stringc& file);

	scene::ISceneManager* SceneManager;
	io::IFileSystem* FileSystem;
	io::IReadFile* File;
	SMesh* Mesh;

	core::array<core::vector3df> Points;
	core::array<core::array<uint32_t> > Indices;
	core::array<core::stringc> UvName;
	core::array<core::array<uint32_t> > UvIndex;
	core::array<core::stringc> DUvName;
	core::array<core::array<uint32_t> > VmPolyPointsIndex;
	core::array<core::array<core::vector2df> > VmCoordsIndex;

	core::array<uint16_t> MaterialMapping;
	core::array<core::array<core::vector2df> > TCoords;
	core::array<tLWOMaterial*> Materials;
	core::array<core::stringc> Images;
	uint8_t FormatVersion;
};

} // end namespace scene
} // end namespace irr
#endif // _IRR_COMPILE_WITH_LWO_LOADER_

#endif
