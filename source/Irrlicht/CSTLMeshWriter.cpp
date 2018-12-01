// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_STL_WRITER_

#include "CSTLMeshWriter.h"
#include "os.h"
#include "IMesh.h"
#include "IMeshBuffer.h"
#include "ISceneManager.h"
#include "IMeshCache.h"
#include "IWriteFile.h"
#include "IFileSystem.h"
#include <sstream>

namespace irr
{
namespace scene
{

CSTLMeshWriter::CSTLMeshWriter(scene::ISceneManager* smgr)
	: SceneManager(smgr)
{
	#ifdef _DEBUG
	setDebugName("CSTLMeshWriter");
	#endif

	if (SceneManager)
		SceneManager->grab();
}


CSTLMeshWriter::~CSTLMeshWriter()
{
	if (SceneManager)
		SceneManager->drop();
}

//! writes a mesh
bool CSTLMeshWriter::writeAsset(io::IWriteFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
{
    if (!_override)
        getDefaultOverride(_override);

    SAssetWriteContext ctx{_params, _file};

    const ICPUMesh* mesh =
#   ifndef _DEBUG
        static_cast<const ICPUMesh*>(_params.rootAsset);
#   else
        dynamic_cast<const ICPUMesh*>(_params.rootAsset);
#   endif
    assert(mesh);

    io::IWriteFile* file = _override->getOutputFile(_file, ctx, {mesh, 0u});

	if (!file)
		return false;

	os::Printer::log("Writing mesh", file->getFileName().c_str());

    const asset::E_WRITER_FLAGS flags = _override->getAssetWritingFlags(ctx, mesh, 0u);
	if (flags & asset::EWF_BINARY)
		return writeMeshBinary(file, mesh);
	else
		return writeMeshASCII(file, mesh);
}

namespace
{
template <class I>
inline void writeFacesBinary(ICPUMeshBuffer* buffer, const bool& noIndices, io::IWriteFile* file, scene::E_VERTEX_ATTRIBUTE_ID _colorVaid)
{
    bool hasColor = buffer->getMeshDataAndFormat()->getMappedBuffer(_colorVaid);
    const scene::E_COMPONENT_TYPE colorType = buffer->getMeshDataAndFormat()->getAttribType(_colorVaid);

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
            if (scene::isNativeInteger(colorType))
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


        const core::plane3df plane(v[0].getAsVector3df(),v[1].getAsVector3df(),v[2].getAsVector3df());
        file->write(&plane.Normal, 12);
        file->write(v+0, 12);
        file->write(v+1, 12);
        file->write(v+2, 12);
        file->write(&color, 2); // saving color using non-standard VisCAM/SolidView trick
    }
}
}

bool CSTLMeshWriter::writeMeshBinary(io::IWriteFile* file, const scene::ICPUMesh* mesh)
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
		ICPUMeshBuffer* buffer = mesh->getMeshBuffer(i);
		if (buffer&&buffer->getMeshDataAndFormat())
		{
			E_INDEX_TYPE type = buffer->getIndexType();
			if (!buffer->getMeshDataAndFormat()->getIndexBuffer())
                type = EIT_UNKNOWN;
			if (type==EIT_16BIT)
            {
                writeFacesBinary<uint16_t>(buffer, false, file, scene::EVAI_ATTR1);
            }
			else if (type==EIT_32BIT)
            {
                writeFacesBinary<uint32_t>(buffer, false, file, scene::EVAI_ATTR1);
            }
			else
            {
                writeFacesBinary<uint16_t>(buffer, true, file, scene::EVAI_ATTR1); //template param doesn't matter if there's no indices
            }
		}
	}
	return true;
}


bool CSTLMeshWriter::writeMeshASCII(io::IWriteFile* file, const scene::ICPUMesh* mesh)
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
		ICPUMeshBuffer* buffer = mesh->getMeshBuffer(i);
		if (buffer&&buffer->getMeshDataAndFormat())
		{
			E_INDEX_TYPE type = buffer->getIndexType();
			if (!buffer->getMeshDataAndFormat()->getIndexBuffer())
                type = EIT_UNKNOWN;
			const uint32_t indexCount = buffer->getIndexCount();
			if (type==EIT_16BIT)
			{
                //os::Printer::log("Writing mesh with 16bit indices");
                for (uint32_t j=0; j<indexCount; j+=3)
                {
                    writeFaceText(file,
                        buffer->getPosition(((uint16_t*)buffer->getIndices())[j]).getAsVector3df(),
                        buffer->getPosition(((uint16_t*)buffer->getIndices())[j+1]).getAsVector3df(),
                        buffer->getPosition(((uint16_t*)buffer->getIndices())[j+2]).getAsVector3df()
                    );
                }
			}
			else if (type==EIT_32BIT)
			{
                //os::Printer::log("Writing mesh with 32bit indices");
                for (uint32_t j=0; j<indexCount; j+=3)
                {
                    writeFaceText(file,
                        buffer->getPosition(((uint32_t*)buffer->getIndices())[j]).getAsVector3df(),
                        buffer->getPosition(((uint32_t*)buffer->getIndices())[j+1]).getAsVector3df(),
                        buffer->getPosition(((uint32_t*)buffer->getIndices())[j+2]).getAsVector3df()
                    );
                }
			}
			else
            {
                //os::Printer::log("Writing mesh with no indices");
                for (uint32_t j=0; j<indexCount; j+=3)
                {
                    writeFaceText(file,
                        buffer->getPosition(j).getAsVector3df(),
                        buffer->getPosition(j+1).getAsVector3df(),
                        buffer->getPosition(j+2).getAsVector3df()
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


void CSTLMeshWriter::getVectorAsStringLine(const core::vector3df& v, core::stringc& s) const
{
    std::ostringstream tmp;
    tmp << v.X << " " << v.Y << " " << v.Z << "\n";
    s = core::stringc(tmp.str().c_str());
}


void CSTLMeshWriter::writeFaceText(io::IWriteFile* file,
		const core::vector3df& v1,
		const core::vector3df& v2,
		const core::vector3df& v3)
{
	core::stringc tmp;
	file->write("facet normal ",13);
	getVectorAsStringLine(core::plane3df(v1,v2,v3).Normal, tmp);
	file->write(tmp.c_str(),tmp.size());
	file->write("  outer loop\n",13);
	file->write("    vertex ",11);
	getVectorAsStringLine(v1, tmp);
	file->write(tmp.c_str(),tmp.size());
	file->write("    vertex ",11);
	getVectorAsStringLine(v2, tmp);
	file->write(tmp.c_str(),tmp.size());
	file->write("    vertex ",11);
	getVectorAsStringLine(v3, tmp);
	file->write(tmp.c_str(),tmp.size());
	file->write("  endloop\n",10);
	file->write("endfacet\n",9);
}

} // end namespace
} // end namespace

#endif

