#ifndef __C2B_PRINT_H_INCLUDED__
#define __C2B_PRINT_H_INCLUDED__

namespace irr 
{ 
	namespace core
	{
		class ICPUBuffer;
	}
	namespace scene 
	{
		class ICPUMesh;
		class ICPUMeshBuffer;
		template<typename> class IMeshDataFormatDesc;
	}
}
void printFullMeshInfo(const irr::scene::ICPUMesh* _mesh, size_t _indent = 0u);
void printMeshInfo(const irr::scene::ICPUMesh* _mesh, size_t _indent);
void printMeshBufferInfo(const irr::scene::ICPUMeshBuffer* _buf, size_t _indent);
void printDescInfo(const irr::scene::IMeshDataFormatDesc<irr::core::ICPUBuffer>* _desc, size_t _indent);


#endif