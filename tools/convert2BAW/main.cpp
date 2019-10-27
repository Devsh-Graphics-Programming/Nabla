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

#define _IRR_STATIC_LIB_

#include <irrlicht.h>
#include <SMesh.h>
#include <SSkinMeshBuffer.h>
#include <IMeshBuffer.h>
#include "../source/Irrlicht/CBAWMeshWriter.h"
#include "../source/Irrlicht/CSkinnedMesh.h"
#include <vector>
#include <cstdlib>
#include <chrono>

#include "print.h"

// Usage: convert2BAW [-i [list of input files delimited with spaces]] [-o [list of output files delimited with spaces]]
//			[-rel <dir>] [-pwd <password>] [-optmesh <{ error metric settings threes delimited with commas }>]
// Options:
// -i [list of input files]
// -o [list of output files]
//	Output files must be of *.baw extension.
// -rel <path>
//	Directory to which textures in output mesh files will be relative.
// -pwd <password>
//	Password string consisting of only hex digits. Must be 32 characters long.
// -info
// Prints mesh info to stdout.
// -optmesh <settings>
//	If passed - mesh will be optimized before export. The option comes along with error metrics settings:
//	Settings must be enclosed with curly (i.e. {}) braces and grouped in threes. Threes must be delimited with commas. Order of threes is irrelevant.
//	Elements of each group of three must be delimited with spaces and must come with strict order: atrribute-id epsilon cmp-method
//	Attribute-id must be integer in range [0; 15]. Epsilon is floating point number. Cmp-method must be single character and one of: A - angles, Q - quaternions, P - positions (lower-case chars are also accepted)

//Example:
//	convert2BAW -i somefile.obj someotherfile.x -o f1.baw f2.baw -rel /home/me/assets/ -pwd deadbeefbaadf00d0badcafefeeee997 -optmesh { 0 0.02 P, 3 0.003 A }


using namespace irr;

enum E_GATHER_TARGET
{
	EGT_UNDEFINED = 0,
	EGT_INPUTS,
	EGT_OUTPUTS
};

static bool checkHex(const char* _str);
//! Input must be 32 bytes long. Output buffer must be at least 16 bytes.
static void hexStrToIntegers(const char* _input, unsigned char* _out);
static uint8_t hexCharToUint8(char _c);
static bool optMesh(scene::ICPUMesh* _mesh, const scene::IMeshManipulator* _manip, const scene::IMeshManipulator::SErrorMetric* _errMetrics);

int main(int _optCnt, char** _options)
{
	--_optCnt;
	++_options;

	irr::SIrrlichtCreationParameters params;
	params.DriverType = video::EDT_OPENGL;
	params.WindowSize = core::dimension2d<uint32_t>(128, 128);
	IrrlichtDevice* device = createDeviceEx(params);

	if (!device)
		return 1;

	scene::ISceneManager* const smgr = device->getSceneManager();
	io::IFileSystem* const fs = device->getFileSystem();
	scene::CBAWMeshWriter* const writer = dynamic_cast<scene::CBAWMeshWriter*>(smgr->createMeshWriter(irr::scene::EMWT_BAW));
	scene::IMeshManipulator* const meshManip = smgr->getMeshManipulator();

	std::vector<const char*> inNames;
	std::vector<const char*> outNames;

	E_GATHER_TARGET gatherWhat = EGT_UNDEFINED;
	bool usePwd = 0;
	bool optimizeMesh = 0;
	bool printInfo = 0;
	scene::CBAWMeshWriter::WriteProperties properties;
	scene::IMeshManipulator::SErrorMetric errMetrics[16];

	srand(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	for (size_t i = 0u; i < 16u; ++i)
    {
		reinterpret_cast<uint8_t*>(properties.initializationVector)[i] = rand()%256;
		reinterpret_cast<uint8_t*>(properties.initializationVector)[i] ^= std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }

	for (size_t idx = 0u; idx < _optCnt; ++idx)
	{
		if (_options[idx][0] == '-')
		{
			if (core::equalsIgnoreCase("i", _options[idx]+1))
				gatherWhat = EGT_INPUTS;
			else if (core::equalsIgnoreCase("o", _options[idx]+1))
				gatherWhat = EGT_OUTPUTS;
			else if (idx+1 != _optCnt && core::equalsIgnoreCase("pwd", _options[idx]+1))
			{
				++idx;
				gatherWhat = EGT_UNDEFINED;
				if (core::length(_options[idx]) != 32)
				{
					printf("Password length must be 32! Ignored - password not set.\n");
					continue;
				}
				if (!checkHex(_options[idx]))
				{
					printf("Password must consist of only hex digits! Ignore - password not set.\n");
					continue;
				}
				hexStrToIntegers(_options[idx], properties.encryptionPassPhrase);
				usePwd = 1;
				continue;
			}
			else if (idx+1 != _optCnt && core::equalsIgnoreCase("rel", _options[idx]+1))
			{
				++idx;
				gatherWhat = EGT_UNDEFINED;
				properties.relPath = _options[idx];
				continue;
			}
			else if (core::equalsIgnoreCase("info", _options[idx]+1))
			{
				gatherWhat = EGT_UNDEFINED;
				printInfo = 1;
				continue;
			}
			else if (idx+1 != _optCnt && core::equalsIgnoreCase("optmesh", _options[idx]+1))
			{
				gatherWhat = EGT_UNDEFINED;
				optimizeMesh = 1;

				++idx;

				if (_options[idx][0] != '{')
				{
					printf("List of error metrics settings should be enclosed with curly braces {}.\n");
					continue;
				}

				std::string s;
				do
				{
                    s += std::string(_options[idx]) + ' ';

				} while (!strstr(_options[idx++], "}"));
				--idx;

				const size_t cnt = std::count(s.begin(), s.end(), ',')+1;
				std::transform(s.cbegin(), s.cend(), s.begin(), [](char c) { return (c == '{' || c == '}' || c == ',') ? ' ' : c; });

				std::stringstream ss(s);
				for (size_t i = 0u; i < cnt; ++i)
				{
					size_t vaid{};
					ss >> vaid;
					float eps{};
					ss >> eps;
					errMetrics[vaid].epsilon = core::vectorSIMDf(eps);
					char method{};
					ss >> method;

					method = tolower(method);
					switch (method)
					{
					case 'a':
						errMetrics[vaid].method = scene::IMeshManipulator::EEM_ANGLES;
						break;
					case 'q':
						errMetrics[vaid].method = scene::IMeshManipulator::EEM_QUATERNION;
						break;
					case 'p':
						errMetrics[vaid].method = scene::IMeshManipulator::EEM_POSITIONS;
						break;
					}
				}

				continue;
			}
			else
			{
				gatherWhat = EGT_UNDEFINED;
				printf("Ignored unrecognized option \"%s\".\n", _options[idx]);
			}
			continue;
		}

		switch (gatherWhat)
		{
		case EGT_INPUTS:
			inNames.push_back(_options[idx]);
			break;
		case EGT_OUTPUTS:
			if (!core::hasFileExtension(_options[idx], "baw"))
			{
				printf("Output filename must be of 'baw' extension. Ignored.\n");
				break;
			}
			outNames.push_back(_options[idx]);
			break;
		default:
			printf("Ignored an input \"%s\".\n", _options[idx]);
			break;
		}
	}

	if (inNames.size() != outNames.size())
	{
		printf("Fatal error. Amounts of input and output filenames doesn't match. Exiting.\n");
        writer->drop();
        device->drop();
		return 1;
	}

	for (size_t i = 0u; i < inNames.size(); ++i)
	{
		scene::ICPUMesh* inmesh = smgr->getMesh(inNames[i]);
		if (!inmesh)
		{
			printf("Could not load mesh %s.\n", inNames[i]);
			continue;
		}
		io::IWriteFile* outfile = fs->createAndWriteFile(outNames[i]);
		if (!outfile)
		{
			printf("Could not create/open file %s.\n", outNames[i]);
            smgr->getMeshCache()->removeMesh(inmesh);
			continue;
		}
		if (optimizeMesh && !optMesh(inmesh, meshManip, errMetrics))
		{
			printf("Could not optimize mesh %s. Mesh not exported!\n", inNames[i]);
			smgr->getMeshCache()->removeMesh(inmesh);
            outfile->drop();
			continue;
		}

        if (printInfo)
        {
            printf("%s INFO:\n", inNames[i]);
            printFullMeshInfo(stdout, inmesh);
        }

		if (usePwd)
			writer->writeMesh(outfile, inmesh, properties);
		else
			writer->writeMesh(outfile, inmesh, scene::EMWF_WRITE_COMPRESSED);

        smgr->getMeshCache()->removeMesh(inmesh);
		outfile->drop();
	}
	writer->drop();
	device->drop();

	return 0;
}

static bool checkHex(const char* _str)
{
	const size_t len = core::length(_str);

	for (size_t i = 0u; i < len; ++i)
	{
		const char c = tolower(_str[i]);
		if ((c < 'a' || c > 'f') && !isdigit(c))
			return false;
	}
	return true;
}

static void hexStrToIntegers(const char* _input, unsigned char* _out)
{
	for (size_t i = 0; i < 32u; i += 2u, ++_out)
	{
		const uint8_t val = hexCharToUint8(_input[i]) + hexCharToUint8(_input[i+1])*16;
		*_out = val;
	}
}

static uint8_t hexCharToUint8(char _c)
{
	if (isdigit(_c))
		return _c - '0';
	return tolower(_c) - 'a' + 10;
}

//static bool optMesh(scene::ICPUMesh* _mesh, const scene::IMeshManipulator* _manip, const scene::IMeshManipulator::SErrorMetric* _errMetrics)
//{
//	std::vector<scene::ICPUMeshBuffer*> buffers;
//
//	bool status = true;
//	size_t c = _mesh->getMeshBufferCount();
//	for (size_t i = 0u; i < _mesh->getMeshBufferCount(); ++i)
//	{
//		scene::ICPUMeshBuffer* optdBuf = _manip->createOptimizedMeshBuffer(_mesh->getMeshBuffer(i), _errMetrics);
//		if (!optdBuf)
//		{
//			status = false;
//			break;
//		}
//		buffers.push_back(optdBuf);
//	}
//
//	if (status)
//	{
//		_mesh->clearMeshBuffers();
//		for (scene::ICPUMeshBuffer* b : buffers)
//			_mesh->addMeshBuffer(b);
//	}
//
//	for (const scene::ICPUMeshBuffer* b : buffers)
//		b->drop();
//
//	return status;
//}
template<typename MeshT, typename MeshBufT>
static bool _optMesh(MeshT* _mesh, const scene::IMeshManipulator* _manip, const scene::IMeshManipulator::SErrorMetric* _errMetrics)
{
    std::vector<scene::ICPUMeshBuffer*> buffers;

    bool status = true;
    size_t c = _mesh->getMeshBufferCount();
    for (size_t i = 0u; i < _mesh->getMeshBufferCount(); ++i)
    {
        scene::ICPUMeshBuffer* optdBuf = _manip->createOptimizedMeshBuffer(_mesh->getMeshBuffer(i), _errMetrics);
        if (!optdBuf)
        {
            status = false;
            break;
        }
        buffers.push_back(optdBuf);
    }

    if (status)
    {
        _mesh->clearMeshBuffers();
        for (scene::ICPUMeshBuffer* b : buffers)
            if (MeshBufT* bb = dynamic_cast<MeshBufT*>(b))
            _mesh->addMeshBuffer(bb);
    }

    for (const scene::ICPUMeshBuffer* b : buffers)
        b->drop();

    return status;
}
static bool optMesh(scene::ICPUMesh* _mesh, const scene::IMeshManipulator* _manip, const scene::IMeshManipulator::SErrorMetric* _errMetrics)
{
    if (scene::CCPUSkinnedMesh* m = dynamic_cast<scene::CCPUSkinnedMesh*>(_mesh))
        return _optMesh<scene::CCPUSkinnedMesh, scene::SCPUSkinMeshBuffer>(m, _manip, _errMetrics);
    else if (scene::SCPUMesh* m = dynamic_cast<scene::SCPUMesh*>(_mesh))
        return _optMesh<scene::SCPUMesh, scene::ICPUMeshBuffer>(m, _manip, _errMetrics);
    return false;
}
