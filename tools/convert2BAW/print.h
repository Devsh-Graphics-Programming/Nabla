# Copyright(c) 2019 DevSH Graphics Programming Sp.z O.O.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissionsand
# limitations under the License.

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