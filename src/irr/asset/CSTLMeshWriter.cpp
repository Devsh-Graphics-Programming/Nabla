// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "irr/core/core.h"

#ifdef _IRR_COMPILE_WITH_STL_WRITER_

#include "CSTLMeshWriter.h"

#include "os.h"
#include "IWriteFile.h"
#include "IFileSystem.h"
#include "ISceneManager.h"

namespace irr
{
namespace asset
{

CSTLMeshWriter::CSTLMeshWriter()
{
	#ifdef _IRR_DEBUG
	setDebugName("CSTLMeshWriter");
	#endif
}


CSTLMeshWriter::~CSTLMeshWriter()
{
}

//! writes a mesh
bool CSTLMeshWriter::writeAsset(io::IWriteFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
{
    if (!_override)
        getDefaultOverride(_override);

    SAssetWriteContext ctx{_params, _file};

    const asset::ICPUMesh* mesh =
#   ifndef _IRR_DEBUG
        static_cast<const asset::ICPUMesh*>(_params.rootAsset);
#   else
        dynamic_cast<const asset::ICPUMesh*>(_params.rootAsset);
#   endif
    assert(mesh);

    io::IWriteFile* file = _override->getOutputFile(_file, ctx, {mesh, 0u});

	if (!file)
		return false;

	os::Printer::log("Writing mesh", file->getFileName().c_str());

    const asset::E_WRITER_FLAGS flags = _override->getAssetWritingFlags(ctx, mesh, 0u);
	if (flags & asset::EWF_BINARY)
		return writeMeshBinary(file, mesh, _params);
	else
		return writeMeshASCII(file, mesh, _params);
}

namespace
{
template <class I>
inline void writeFacesBinary(asset::ICPUMeshBuffer* buffer, const bool& noIndices, io::IWriteFile* file, asset::E_VERTEX_ATTRIBUTE_ID _colorVaid, const irr::asset::IAssetWriter::SAssetWriteParams& _params)
{
    bool hasColor = buffer->getMeshDataAndFormat()->getMappedBuffer(_colorVaid);
    const asset::E_FORMAT colorType = buffer->getMeshDataAndFormat()->getAttribFormat(_colorVaid);

    const uint32_t indexCount = buffer->getIndexCount();
    for (uint32_t j = 0u; j < indexCount; j += 3u)
    {
        I idx[3];
        for (uint32_t i = 0u; i < 3u; ++i)
        {
            if (noIndices)
                idx[i] = j + i;
            else
                idx[i] = ((I*)buffer->getIndices())[j + i];
        }

        core::vectorSIMDf v[3];
        for (uint32_t i = 0u; i < 3u; ++i)
            v[i] = buffer->getPosition(idx[i]);

        uint16_t color = 0u;
        if (hasColor)
        {
            if (asset::isIntegerFormat(colorType))
            {
                uint32_t res[4];
                for (uint32_t i = 0u; i < 3u; ++i)
                {
                    uint32_t d[4];
                    buffer->getAttribute(d, _colorVaid, idx[i]);
                    res[0] += d[0]; res[1] += d[1]; res[2] += d[2];
                }
                color = video::RGB16(res[0]/3, res[1]/3, res[2]/3);
            }
            else
            {
                core::vectorSIMDf res;
                for (uint32_t i = 0u; i < 3u; ++i)
                {
                    core::vectorSIMDf d;
                    buffer->getAttribute(d, _colorVaid, idx[i]);
                    res += d;
                }
                res /= 3.f;
                color = video::RGB16(res.X, res.Y, res.Z);
            }
        }

		core::vectorSIMDf normal = core::plane3dSIMDf(v[0], v[1], v[2]).getNormal();
		core::vectorSIMDf vertex1 = v[2];
		core::vectorSIMDf vertex2 = v[1];
		core::vectorSIMDf vertex3 = v[0];

		auto flipVectors = [&]()
		{
			vertex1.X = -vertex1.X;
			vertex2.X = -vertex2.X;
			vertex3.X = -vertex3.X;
			normal = core::plane3dSIMDf(vertex1, vertex2, vertex3).getNormal();
		};

		if (!(_params.flags & E_WRITER_FLAGS::EWF_MESH_IS_RIGHT_HANDED))
			flipVectors();

        file->write(&normal, 12);
        file->write(&vertex1, 12);
        file->write(&vertex2, 12);
        file->write(&vertex3, 12);
        file->write(&color, 2); // saving color using non-standard VisCAM/SolidView trick
    }
}
}

bool CSTLMeshWriter::writeMeshBinary(io::IWriteFile* file, const asset::ICPUMesh* mesh, const SAssetWriteParams& _params)
{
	// write STL MESH header
    const char headerTxt[] = "Irrlicht-baw Engine";
    constexpr size_t HEADER_SIZE = 80u;

	file->write(headerTxt,sizeof(headerTxt));
	const core::stringc name(io::IFileSystem::getFileBasename(file->getFileName(),false));
	const int32_t sizeleft = HEADER_SIZE - sizeof(headerTxt) - name.size();
	if (sizeleft<0)
		file->write(name.c_str(), HEADER_SIZE - sizeof(headerTxt));
	else
	{
		const char buf[80] = {0};
		file->write(name.c_str(),name.size());
		file->write(buf,sizeleft);
	}
	uint32_t facenum = 0;
	for (uint32_t j=0; j<mesh->getMeshBufferCount(); ++j)
		facenum += mesh->getMeshBuffer(j)->getIndexCount()/3;
	file->write(&facenum,4);

	// write mesh buffers

	for (uint32_t i=0; i<mesh->getMeshBufferCount(); ++i)
	{
		asset::ICPUMeshBuffer* buffer = mesh->getMeshBuffer(i);
		if (buffer&&buffer->getMeshDataAndFormat())
		{
            asset::E_INDEX_TYPE type = buffer->getIndexType();
			if (!buffer->getMeshDataAndFormat()->getIndexBuffer())
                type = asset::EIT_UNKNOWN;
			if (type== asset::EIT_16BIT)
            {
                writeFacesBinary<uint16_t>(buffer, false, file, asset::EVAI_ATTR1, _params);
            }
			else if (type== asset::EIT_32BIT)
            {
                writeFacesBinary<uint32_t>(buffer, false, file, asset::EVAI_ATTR1, _params);
            }
			else
            {
                writeFacesBinary<uint16_t>(buffer, true, file, asset::EVAI_ATTR1, _params); //template param doesn't matter if there's no indices
            }
		}
	}
	return true;
}


bool CSTLMeshWriter::writeMeshASCII(io::IWriteFile* file, const asset::ICPUMesh* mesh, const SAssetWriteParams& _params)
{
	// write STL MESH header
    const char headerTxt[] = "Irrlicht-baw Engine ";

	file->write("solid ",6);
    file->write(headerTxt, sizeof(headerTxt)-1);
	const core::stringc name(io::IFileSystem::getFileBasename(file->getFileName(), false));
	file->write(name.c_str(), name.size());
	file->write("\n", 1);

	// write mesh buffers

	for (uint32_t i=0; i<mesh->getMeshBufferCount(); ++i)
	{
		asset::ICPUMeshBuffer* buffer = mesh->getMeshBuffer(i);
		if (buffer&&buffer->getMeshDataAndFormat())
		{
            asset::E_INDEX_TYPE type = buffer->getIndexType();
			if (!buffer->getMeshDataAndFormat()->getIndexBuffer())
                type = asset::EIT_UNKNOWN;
			const uint32_t indexCount = buffer->getIndexCount();
			if (type==asset::EIT_16BIT)
			{
                //os::Printer::log("Writing mesh with 16bit indices");
                for (uint32_t j=0; j<indexCount; j+=3)
                {
                    writeFaceText(file,
                        buffer->getPosition(((uint16_t*)buffer->getIndices())[j]),
                        buffer->getPosition(((uint16_t*)buffer->getIndices())[j+1]),
                        buffer->getPosition(((uint16_t*)buffer->getIndices())[j+2]),
						_params
                    );
                }
			}
			else if (type==asset::EIT_32BIT)
			{
                //os::Printer::log("Writing mesh with 32bit indices");
                for (uint32_t j=0; j<indexCount; j+=3)
                {
                    writeFaceText(file,
                        buffer->getPosition(((uint32_t*)buffer->getIndices())[j]),
                        buffer->getPosition(((uint32_t*)buffer->getIndices())[j+1]),
                        buffer->getPosition(((uint32_t*)buffer->getIndices())[j+2]),
						_params
                    );
                }
			}
			else
            {
                //os::Printer::log("Writing mesh with no indices");
                for (uint32_t j=0; j<indexCount; j+=3)
                {
                    writeFaceText(file,
                        buffer->getPosition(j),
                        buffer->getPosition(j+1ul),
                        buffer->getPosition(j+2ul),
						_params
                    );
                }
            }
			file->write("\n",1);
		}
	}

	file->write("endsolid ",9);
    file->write(headerTxt, sizeof(headerTxt)-1);
	file->write(name.c_str(),name.size());

	return true;
}


void CSTLMeshWriter::getVectorAsStringLine(const core::vectorSIMDf& v, core::stringc& s) const
{
    std::ostringstream tmp;
    tmp << v.X << " " << v.Y << " " << v.Z << "\n";
    s = core::stringc(tmp.str().c_str());
}


void CSTLMeshWriter::writeFaceText(io::IWriteFile* file,
		const core::vectorSIMDf& v1,
		const core::vectorSIMDf& v2,
		const core::vectorSIMDf& v3,
	    const SAssetWriteParams& _params)
{
	core::vectorSIMDf vertex1 = v3;
	core::vectorSIMDf vertex2 = v2;
	core::vectorSIMDf vertex3 = v1;
	core::vectorSIMDf normal = core::plane3dSIMDf(vertex1, vertex2, vertex3).getNormal();
	core::stringc tmp;

	auto flipVectors = [&]()
	{
		vertex1.X = -vertex1.X;
		vertex2.X = -vertex2.X;
		vertex3.X = -vertex3.X;
		normal = core::plane3dSIMDf(vertex1, vertex2, vertex3).getNormal();
	};

	if (!(_params.flags & E_WRITER_FLAGS::EWF_MESH_IS_RIGHT_HANDED))
		flipVectors();

	file->write("facet normal ",13);
	getVectorAsStringLine(normal, tmp);
	file->write(tmp.c_str(),tmp.size());
	file->write("  outer loop\n",13);
	file->write("    vertex ",11);
	getVectorAsStringLine(vertex1, tmp);
	file->write(tmp.c_str(),tmp.size());
	file->write("    vertex ",11);
	getVectorAsStringLine(vertex2, tmp);
	file->write(tmp.c_str(),tmp.size());
	file->write("    vertex ",11);
	getVectorAsStringLine(vertex3, tmp);
	file->write(tmp.c_str(),tmp.size());
	file->write("  endloop\n",10);
	file->write("endfacet\n",9);
}

} // end namespace
} // end namespace

#endif
