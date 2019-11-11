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

#include "print.h"

#include <IMesh.h>
#include <IMeshBuffer.h>

using namespace irr;

static std::string idxTypeToStr(scene::E_INDEX_TYPE _it) 
{
	switch (_it)
	{
	case scene::EIT_16BIT: return "EIT_16BIT";
	case scene::EIT_32BIT: return "EIT_32BIT";
	case scene::EIT_UNKNOWN: return "EIT_UNKNOWN";
	}
	return "";
}
static std::string primitiveTypeToStr(scene::E_PRIMITIVE_TYPE _pt)
{
	switch (_pt)
	{
	case scene::EPT_POINTS: return "EPT_POINTS";
	case scene::EPT_LINE_STRIP: return "EPT_LINE_STRIP";
	case scene::EPT_LINE_LOOP: return "EPT_LINE_LOOP";
	case scene::EPT_LINES: return "EPT_LINES";
	case scene::EPT_TRIANGLE_STRIP: return "EPT_TRIANGLE_STRIP";
	case scene::EPT_TRIANGLE_FAN: return "EPT_TRIANGLE_FAN";
	case scene::EPT_TRIANGLES: return "EPT_TRIANGLES";
	}
	return "";
}
static std::string cmpntTypeToStr(scene::E_COMPONENT_TYPE _cp)
{
	using namespace scene;
	switch (_cp)
	{
	case ECT_FLOAT: return "ECT_FLOAT";
	case ECT_HALF_FLOAT: return "ECT_HALF_FLOAT";
	case ECT_DOUBLE_IN_FLOAT_OUT: return "ECT_DOUBLE_IN_FLOAT_OUT";
	case ECT_UNSIGNED_INT_10F_11F_11F_REV: return "ECT_UNSIGNED_INT_10F_11F_11F_REV";
	case ECT_NORMALIZED_INT_2_10_10_10_REV: return "ECT_NORMALIZED_INT_2_10_10_10_REV";
	case ECT_NORMALIZED_UNSIGNED_INT_2_10_10_10_REV: return "ECT_NORMALIZED_UNSIGNED_INT_2_10_10_10_REV";
	case ECT_NORMALIZED_BYTE: return "ECT_NORMALIZED_BYTE";
	case ECT_NORMALIZED_UNSIGNED_BYTE: return "ECT_NORMALIZED_UNSIGNED_BYTE";
	case ECT_NORMALIZED_SHORT: return "ECT_NORMALIZED_SHORT";
	case ECT_NORMALIZED_UNSIGNED_SHORT: return "ECT_NORMALIZED_UNSIGNED_SHORT";
	case ECT_NORMALIZED_INT: return "ECT_NORMALIZED_INT";
	case ECT_NORMALIZED_UNSIGNED_INT: return "ECT_NORMALIZED_UNSIGNED_INT";
	case ECT_INT_2_10_10_10_REV: return "ECT_INT_2_10_10_10_REV";
	case ECT_UNSIGNED_INT_2_10_10_10_REV: return "ECT_UNSIGNED_INT_2_10_10_10_REV";
	case ECT_BYTE: return "ECT_BYTE";
	case ECT_UNSIGNED_BYTE: return "ECT_UNSIGNED_BYTE";
	case ECT_SHORT: return "ECT_SHORT";
	case ECT_UNSIGNED_SHORT: return "ECT_UNSIGNED_SHORT";
	case ECT_INT: return "ECT_INT";
	case ECT_UNSIGNED_INT: return "ECT_UNSIGNED_INT";
	case ECT_INTEGER_INT_2_10_10_10_REV: return "ECT_INTEGER_INT_2_10_10_10_REV";
	case ECT_INTEGER_UNSIGNED_INT_2_10_10_10_REV: return "ECT_INTEGER_UNSIGNED_INT_2_10_10_10_REV";
	case ECT_INTEGER_BYTE: return "ECT_INTEGER_BYTE";
	case ECT_INTEGER_UNSIGNED_BYTE: return "ECT_INTEGER_UNSIGNED_BYTE";
	case ECT_INTEGER_SHORT: return "ECT_INTEGER_SHORT";
	case ECT_INTEGER_UNSIGNED_SHORT: return "ECT_INTEGER_UNSIGNED_SHORT";
	case ECT_INTEGER_INT: return "ECT_INTEGER_INT";
	case ECT_INTEGER_UNSIGNED_INT: return "ECT_INTEGER_UNSIGNED_INT";
	case ECT_DOUBLE_IN_DOUBLE_OUT: return "ECT_DOUBLE_IN_DOUBLE_OUT";
	}
	return "";
}

void printFullMeshInfo(FILE* _ostream, const irr::scene::ICPUMesh * _mesh, size_t _indent)
{
	const std::string indent(_indent, '\t');

	fprintf(_ostream, "Mesh %p:\n", _mesh);
	printMeshInfo(_ostream, _mesh, _indent+1);
}

void printMeshInfo(FILE* _ostream, const scene::ICPUMesh* _mesh, size_t _indent)
{
	const std::string indent(_indent, '\t');

	for (size_t i = 0u; i < _mesh->getMeshBufferCount(); ++i)
	{
		fprintf(_ostream, "%sMesh buffer %u:\n", indent.c_str(), i);
		printMeshBufferInfo(_ostream, _mesh->getMeshBuffer(i), _indent+1);
	}
}

void printMeshBufferInfo(FILE* _ostream, const scene::ICPUMeshBuffer* _buf, size_t _indent)
{
	const std::string indent(_indent, '\t');

	fprintf(_ostream, "%sindexType: %s\n", indent.c_str(), idxTypeToStr(_buf->getIndexType()).c_str());
	fprintf(_ostream, "%sbaseVertex: %d\n", indent.c_str(), _buf->getBaseVertex());
	fprintf(_ostream, "%sindexCount: %u\n", indent.c_str(), _buf->getIndexCount());
	fprintf(_ostream, "%sindexBufOffset: %u\n", indent.c_str(), _buf->getIndexBufferOffset());
	fprintf(_ostream, "%sinstanceCount: %u\n", indent.c_str(), _buf->getInstanceCount());
	fprintf(_ostream, "%sbaseInstance: %u\n", indent.c_str(), _buf->getBaseInstance());
	fprintf(_ostream, "%sprimitiveType: %s\n", indent.c_str(), primitiveTypeToStr(_buf->getPrimitiveType()).c_str());
	fprintf(_ostream, "%smeshLayout:\n", indent.c_str());
	printDescInfo(_ostream, _buf->getMeshDataAndFormat(), _indent+1);
}

static void printAttributeInfo(FILE* _ostream, const scene::IMeshDataFormatDesc<core::ICPUBuffer>* _desc, scene::E_VERTEX_ATTRIBUTE_ID _vaid, size_t _indent)
{
	const std::string indent(_indent, '\t');

	fprintf(_ostream, "%scompntsPerAttr: %u\n", indent.c_str(), (size_t)_desc->getAttribComponentCount(_vaid));
	fprintf(_ostream, "%sattrType: %s\n", indent.c_str(), cmpntTypeToStr(_desc->getAttribType(_vaid)).c_str());
	fprintf(_ostream, "%sstride: %u\n", indent.c_str(), _desc->getMappedBufferStride(_vaid));
	fprintf(_ostream, "%soffset: %u\n", indent.c_str(), _desc->getMappedBufferOffset(_vaid));
	fprintf(_ostream, "%sdivisor: %u\n", indent.c_str(), _desc->getAttribDivisor(_vaid));
}
void printDescInfo(FILE* _ostream, const scene::IMeshDataFormatDesc<core::ICPUBuffer>* _desc, size_t _indent)
{
	const std::string indent(_indent, '\t');

	for (size_t vaid = 0u; vaid < scene::EVAI_COUNT; ++vaid)
	{
		if (const void* b = _desc->getMappedBuffer((scene::E_VERTEX_ATTRIBUTE_ID)vaid))
		{
			fprintf(_ostream, "%sAttribute %u in buffer %p:\n", indent.c_str(), vaid, b);
			printAttributeInfo(_ostream, _desc, (scene::E_VERTEX_ATTRIBUTE_ID)vaid, _indent+1);
		}
	}
}