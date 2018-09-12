// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CMeshCache.h"
#include "IAnimatedMesh.h"
#include "IMesh.h"

namespace irr
{
namespace scene
{



//! adds a mesh to the list
template<class T>
void CMeshCache<T>::addMesh(const io::path& filename, T* mesh)
{
	mesh->grab();

	MeshEntry<T> e ( filename );
	e.Mesh = mesh;

	auto found = std::lower_bound(Meshes.begin(),Meshes.end(),e);
	Meshes.insert(found,e);
}


//! Removes a mesh from the cache.
template<class T>
void CMeshCache<T>::removeMesh(const T* const mesh)
{
	if ( !mesh )
		return;


	for (auto it=Meshes.begin(); it!=Meshes.end(); it++)
	{
		if (it->Mesh == mesh)
		{
			it->Mesh->drop();
			Meshes.erase(it);
			return;
		}
	}
}


//! Returns amount of loaded meshes
template<class T>
uint32_t CMeshCache<T>::getMeshCount() const
{
	return Meshes.size();
}


//! Returns current number of the mesh
template<class T>
int32_t CMeshCache<T>::getMeshIndex(const T* const mesh) const
{
	for (uint32_t i=0; i<Meshes.size(); ++i)
	{
		if (Meshes[i].Mesh == mesh)
			return (int32_t)i;
	}

	return -1;
}


//! Returns a mesh based on its index number
template<class T>
T* CMeshCache<T>::getMeshByIndex(uint32_t number)
{
	if (number >= Meshes.size())
		return 0;

	return Meshes[number].Mesh;
}


//! Returns a mesh based on its name.
template<class T>
T* CMeshCache<T>::getMeshByName(const io::path& name)
{
	MeshEntry<T> e ( name );
	auto found = std::lower_bound(Meshes.begin(),Meshes.end(),e);
	if (found!=Meshes.end() && !(e<*found))
        return found->Mesh;
    else
        return nullptr;
}


//! Get the name of a loaded mesh, based on its index.
template<class T>
const char* CMeshCache<T>::getMeshName(uint32_t index) const
{
	if (index >= Meshes.size())
		return NULL;

	return Meshes[index].NamedPath.getInternalName().c_str();
}


//! Get the name of a loaded mesh, if there is any.
template<class T>
const char* CMeshCache<T>::getMeshName(const T* const mesh) const
{
	if (!mesh)
		return NULL;

	for (uint32_t i=0; i<Meshes.size(); ++i)
	{
		if (Meshes[i].Mesh == mesh)
			return Meshes[i].NamedPath.getInternalName().c_str();
	}

	return NULL;
}

//! Renames a loaded mesh.
template<class T>
bool CMeshCache<T>::renameMesh(uint32_t index, const io::path& name)
{
	if (index >= Meshes.size())
		return false;

	Meshes[index].NamedPath.setPath(name);
    std::sort(Meshes.begin(),Meshes.end());
	return true;
}


//! Renames a loaded mesh.
template<class T>
bool CMeshCache<T>::renameMesh(const T* const mesh, const io::path& name)
{
	for (uint32_t i=0; i<Meshes.size(); ++i)
	{
		if (Meshes[i].Mesh == mesh)
		{
			Meshes[i].NamedPath.setPath(name);
			std::sort(Meshes.begin(),Meshes.end());
			return true;
		}
	}

	return false;
}


//! returns if a mesh already was loaded
template<class T>
bool CMeshCache<T>::isMeshLoaded(const io::path& name)
{
	return getMeshByName(name) != 0;
}


//! Clears the whole mesh cache, removing all meshes.
template<class T>
void CMeshCache<T>::clear()
{
	for (uint32_t i=0; i<Meshes.size(); ++i)
		Meshes[i].Mesh->drop();

	Meshes.clear();
}

//! Clears all meshes that are held in the mesh cache but not used anywhere else.
template<class T>
void CMeshCache<T>::clearUnusedMeshes()
{
	for (auto it=Meshes.begin(); it!=Meshes.end();)
	{
		if (it->Mesh->getReferenceCount() == 1)
		{
			it->Mesh->drop();
			it = Meshes.erase(it);
		}
		else
            it++;
	}
}
// Instantiate CMeshCache for the supported template type parameters
template class CMeshCache<ICPUMesh>;
template class CMeshCache<IGPUMesh>;


} // end namespace scene
} // end namespace irr

