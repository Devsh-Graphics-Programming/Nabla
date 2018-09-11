// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "IrrCompileConfig.h"
#include "IFileSystem.h"
#include "CConcurrentObjectCache.h"
#include "IReadFile.h"
#include "IWriteFile.h"

#include "os.h"

#include "CAssetManager.h"

#ifdef _IRR_COMPILE_WITH_HALFLIFE_LOADER_
#include "CAnimatedMeshHalfLife.h"
#endif

#ifdef _IRR_COMPILE_WITH_MS3D_LOADER_
#include "CMS3DMeshFileLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_3DS_LOADER_
#include "C3DSMeshFileLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_X_LOADER_
#include "CXMeshFileLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_OCT_LOADER_
#include "COCTLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_CSM_LOADER_
#include "CCSMLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_LMTS_LOADER_
#include "CLMTSMeshFileLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_MY3D_LOADER_
#include "CMY3DMeshFileLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_DMF_LOADER_
#include "CDMFLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_OGRE_LOADER_
#include "COgreMeshFileLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_OBJ_LOADER_
#include "COBJMeshFileLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_B3D_LOADER_
#include "CB3DMeshFileLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_LWO_LOADER_
#include "CLWOMeshFileLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_STL_LOADER_
#include "CSTLMeshFileLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_PLY_LOADER_
#include "CPLYMeshFileLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_BAW_LOADER_
#include "CBAWMeshFileLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_COLLADA_WRITER_
#include "CColladaMeshWriter.h"
#endif

#ifdef _IRR_COMPILE_WITH_STL_WRITER_
#include "CSTLMeshWriter.h"
#endif

#ifdef _IRR_COMPILE_WITH_OBJ_WRITER_
#include "COBJMeshWriter.h"
#endif

#ifdef _IRR_COMPILE_WITH_PLY_WRITER_
#include "CPLYMeshWriter.h"
#endif

#ifdef _IRR_COMPILE_WITH_BAW_WRITER_
#include"CBAWMeshWriter.h"
#endif


#include "CGeometryCreator.h"

namespace irr
{
	namespace asset
	{

		CAssetManager::CAssetManager(io::IFileSystem* fs) : fileSystem(fs)
		{
			if (fileSystem)
				fileSystem->grab();

			//assert(!meshCache.getByKey("test1"));

			ICPUMesh* testPtr = reinterpret_cast<ICPUMesh*>(0xdeadbeefull);
			meshCache.insert("test2", testPtr);
			//assert(meshCache.getByKey("test2")==testPtr);
		}

		/*
		//! constructor
		CSceneManager::CSceneManager(io::IFileSystem* fs)
			: FileSystem(fs),	MeshCache(0),
			IRR_XML_FORMAT_SCENE(L"irr_scene"), IRR_XML_FORMAT_NODE(L"node"), IRR_XML_FORMAT_NODE_ATTR_TYPE(L"type")
		{
			if (FileSystem)
				FileSystem->grab();

			// create geometry creator
			GeometryCreator = new CGeometryCreator();
			MeshManipulator = new CMeshManipulator();

			// create mesh cache if not there already
			MeshCache = new CMeshCache<ICPUMesh>();

			// add file format loaders. add the least commonly used ones first,
			// as these are checked last

			// TODO: now that we have multiple scene managers, these should be
			// shallow copies from the previous manager if there is one.

#ifdef _IRR_COMPILE_WITH_STL_LOADER_
			MeshLoaderList.push_back(new CSTLMeshFileLoader());
#endif
#ifdef _IRR_COMPILE_WITH_PLY_LOADER_ //! ENABLE
			MeshLoaderList.push_back(new CPLYMeshFileLoader(this));
#endif
#ifdef _IRR_COMPILE_WITH_SMF_LOADER_ //! DIE
			MeshLoaderList.push_back(new CSMFMeshFileLoader(Driver));
#endif
#ifdef _IRR_COMPILE_WITH_OCT_LOADER_ //! DIE
			MeshLoaderList.push_back(new COCTLoader(this, FileSystem));
#endif
#ifdef _IRR_COMPILE_WITH_CSM_LOADER_ //! DIE
			MeshLoaderList.push_back(new CCSMLoader(this, FileSystem));
#endif
#ifdef _IRR_COMPILE_WITH_LMTS_LOADER_ //! DIE
			MeshLoaderList.push_back(new CLMTSMeshFileLoader(FileSystem, Driver));
#endif
#ifdef _IRR_COMPILE_WITH_MY3D_LOADER_ //! DIE
			MeshLoaderList.push_back(new CMY3DMeshFileLoader(this, FileSystem));
#endif
#ifdef _IRR_COMPILE_WITH_DMF_LOADER_ //! DIE
			MeshLoaderList.push_back(new CDMFLoader(this, FileSystem));
#endif
#ifdef _IRR_COMPILE_WITH_OGRE_LOADER_ //! DIE
			MeshLoaderList.push_back(new COgreMeshFileLoader(FileSystem, Driver));
#endif
#ifdef _IRR_COMPILE_WITH_HALFLIFE_LOADER_ //! DIE
			MeshLoaderList.push_back(new CHalflifeMDLMeshFileLoader(this));
#endif
#ifdef _IRR_COMPILE_WITH_LWO_LOADER_ //! DIE
			MeshLoaderList.push_back(new CLWOMeshFileLoader(this, FileSystem));
#endif
#ifdef _IRR_COMPILE_WITH_3DS_LOADER_
			MeshLoaderList.push_back(new C3DSMeshFileLoader(this, FileSystem));
#endif
#ifdef _IRR_COMPILE_WITH_X_LOADER_
			MeshLoaderList.push_back(new CXMeshFileLoader(this, FileSystem));
#endif
#ifdef _IRR_COMPILE_WITH_MS3D_LOADER_
			MeshLoaderList.push_back(new CMS3DMeshFileLoader(Driver));
#endif
#ifdef _IRR_COMPILE_WITH_OBJ_LOADER_
			MeshLoaderList.push_back(new COBJMeshFileLoader(this, FileSystem));
#endif
#ifdef _IRR_COMPILE_WITH_B3D_LOADER_
			MeshLoaderList.push_back(new CB3DMeshFileLoader(this));
#endif
#ifdef _IRR_COMPILE_WITH_BAW_LOADER_
			MeshLoaderList.push_back(new CBAWMeshFileLoader(this, FileSystem));
#endif
		}


		//! destructor
		CSceneManager::~CSceneManager()
		{
			if (FileSystem)
				FileSystem->drop();

			if (MeshManipulator)
				MeshManipulator->drop();

			if (GeometryCreator)
				GeometryCreator->drop();

			uint32_t i;
			for (i = 0; i<MeshLoaderList.size(); ++i)
				MeshLoaderList[i]->drop();

			if (MeshCache)
				MeshCache->drop();

			if (Driver)
				Driver->drop();
		}


		//! gets an animateable mesh. loads it if needed. returned pointer must not be dropped.
		ICPUMesh* CSceneManager::getMesh(const io::path& filename)
		{
			ICPUMesh* msh = MeshCache->getMeshByName(filename);
			if (msh)
				return msh;

			io::IReadFile* file = FileSystem->createAndOpenFile(filename);
			msh = getMesh(file);
			if (file)
				file->drop();

			return msh;
		}


		//! gets an animateable mesh. loads it if needed. returned pointer must not be dropped.
		ICPUMesh* CSceneManager::getMesh(io::IReadFile* file)
		{
			if (!file)
			{
				os::Printer::log("Could not load mesh, because file could not be opened", ELL_ERROR);
				return 0;
			}

			io::path name = file->getFileName();
			ICPUMesh* msh = MeshCache->getMeshByName(file->getFileName());
			if (msh)
				return msh;

			// iterate the list in reverse order so user-added loaders can override the built-in ones
			int32_t count = MeshLoaderList.size();
			for (int32_t i = count - 1; i >= 0; --i)
			{
				if (MeshLoaderList[i]->isALoadableFileExtension(name))
				{
					// reset file to avoid side effects of previous calls to createMesh
					file->seek(0);
					msh = MeshLoaderList[i]->createMesh(file);
					if (msh)
					{
						MeshCache->addMesh(file->getFileName(), msh);
						msh->drop();
						break;
					}
				}
			}

			if (!msh)
				os::Printer::log("Could not load mesh, file format seems to be unsupported", file->getFileName().c_str(), ELL_ERROR);
			else
				os::Printer::log("Loaded mesh", file->getFileName().c_str(), ELL_INFORMATION);

			return msh;
		}

		//! Get the active FileSystem
		/** \return Pointer to the FileSystem
		This pointer should not be dropped. See IReferenceCounted::drop() for more information. *
		io::IFileSystem* CSceneManager::getFileSystem()
		{
			return FileSystem;
		}


		//! Adds an external mesh loader.
		void CSceneManager::addExternalMeshLoader(IMeshLoader* externalLoader)
		{
			if (!externalLoader)
				return;

			externalLoader->grab();
			MeshLoaderList.push_back(externalLoader);
		}


		//! Returns the number of mesh loaders supported by Irrlicht at this time
		uint32_t CSceneManager::getMeshLoaderCount() const
		{
			return MeshLoaderList.size();
		}


		//! Retrieve the given mesh loader
		IMeshLoader* CSceneManager::getMeshLoader(uint32_t index) const
		{
			if (index < MeshLoaderList.size())
				return MeshLoaderList[index];
			else
				return 0;
		}



		//! Returns a pointer to the mesh manipulator.
		IMeshManipulator* CSceneManager::getMeshManipulator()
		{
			return MeshManipulator;
		}

		//! Returns an interface to the mesh cache which is shared between all existing scene managers.
		IMeshCache<ICPUMesh>* CSceneManager::getMeshCache()
		{
			return MeshCache;
		}

		//! Returns a mesh writer implementation if available
		IMeshWriter* CSceneManager::createMeshWriter(EMESH_WRITER_TYPE type)
		{
			switch (type)
			{
			case EMWT_STL:
#ifdef _IRR_COMPILE_WITH_STL_WRITER_
				return new CSTLMeshWriter(this);
#else
				return 0;
#endif
			case EMWT_OBJ:
#ifdef _IRR_COMPILE_WITH_OBJ_WRITER_
				return new COBJMeshWriter(this, FileSystem);
#else
				return 0;
#endif

			case EMWT_PLY:
#ifdef _IRR_COMPILE_WITH_PLY_WRITER_
				return new CPLYMeshWriter();
#else
				return 0;
#endif

			case EMWT_BAW:
#ifdef _IRR_COMPILE_WITH_BAW_WRITER_
				return new CBAWMeshWriter(FileSystem);
#else
				return 0;
#endif
			}

			return 0;
		}
		
		// creates a scenemanager
		IAssetManager* createSceneManager(io::IFileSystem* fs)
		{
			return new CAssetManager(fs);
		}*/


	} // end namespace asset
} // end namespace irr

