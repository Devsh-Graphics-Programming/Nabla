#ifndef __C2B_PRINT_H_INCLUDED__
#define __C2B_PRINT_H_INCLUDED__

#include <cstdio>

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
void printFullMeshInfo(FILE* _ostream, const irr::scene::ICPUMesh* _mesh, size_t _indent = 0u);
void printMeshInfo(FILE* _ostream, const irr::scene::ICPUMesh* _mesh, size_t _indent);
void printMeshBufferInfo(FILE* _ostream, const irr::scene::ICPUMeshBuffer* _buf, size_t _indent);
void printDescInfo(FILE* _ostream, const irr::scene::IMeshDataFormatDesc<irr::core::ICPUBuffer>* _desc, size_t _indent);


#endif