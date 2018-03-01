#define _IRR_STATIC_LIB_

#include <irrlicht.h>
#include <CBAWMeshWriter.h>
#include <vector>

using namespace irr;

enum E_GATHER_TARGET
{
	EGT_UNDEFINED = 0,
	EGT_INPUTS,
	EGT_OUTPUTS
};

bool checkHex(const char* _str);
//! Input must be 32 bytes long. Output buffer must be at least 16 bytes.
void hexStrToIntegers(const char* _input, unsigned char* _out);
uint8_t hexCharToUint8(char _c);

int main(int _optCnt, char** _options)
{
	--_optCnt;
	++_options;

	irr::SIrrlichtCreationParameters params;
	params.DriverType = video::EDT_OPENGL;
	params.WindowSize = core::dimension2d<uint32_t>(0, 0);
	IrrlichtDevice* device = createDeviceEx(params);

	if (!device)
		return 1;

	scene::ISceneManager* const smgr = device->getSceneManager();
	io::IFileSystem* const fs = device->getFileSystem();
	scene::CBAWMeshWriter* const writer = dynamic_cast<scene::CBAWMeshWriter*>(smgr->createMeshWriter(irr::scene::EMWT_BAW));

	std::vector<const char*> inNames;
	std::vector<const char*> outNames;

	E_GATHER_TARGET gatherWhat = EGT_UNDEFINED;
	bool usePwd = 0;
	scene::CBAWMeshWriter::WriteProperties properties;
	for (size_t i = 0; i < _optCnt; ++i)
	{
		if (_options[i][0] == '-')
		{
			if (core::equalsIgnoreCase("i", _options[i]+1))
				gatherWhat = EGT_INPUTS;
			else if (core::equalsIgnoreCase("o", _options[i]+1))
				gatherWhat = EGT_OUTPUTS;
			else if (i+1 != _optCnt && core::equalsIgnoreCase("iv", _options[i]+1))
			{
				++i;
				gatherWhat = EGT_UNDEFINED;
				if (core::length(_options[i]) != 32)
				{
					printf("Initialization vector length must be 32! Ignored - IV not set.\n");
					continue;
				}
				if (!checkHex(_options[i]))
				{
					printf("Initialization vector must consist of only hex digits! Ignore - IV not set.\n");
					continue;
				}
				hexStrToIntegers(_options[i], properties.initializationVector);
				continue;
			}
			else if (i+1 != _optCnt && core::equalsIgnoreCase("pwd", _options[i]+1))
			{
				++i;
				gatherWhat = EGT_UNDEFINED;
				if (core::length(_options[i]) != 32)
				{
					printf("Password length must be 32! Ignored - password not set.\n");
					continue;
				}
				if (!checkHex(_options[i]))
				{
					printf("Password must consist of only hex digits! Ignore - password not set.\n");
					continue;
				}
				hexStrToIntegers(_options[i], properties.encryptionPassPhrase);
				usePwd = 1;
				continue;
			}
			else if (i+1 != _optCnt && core::equalsIgnoreCase("rel", _options[i]+1))
			{
				++i;
				gatherWhat = EGT_UNDEFINED;
				properties.relPath = _options[i];
				continue;
			}
			else
			{
				gatherWhat = EGT_UNDEFINED;
				printf("Ignored unrecognized option \"%s\".\n", _options[i]);
			}
			continue;
		}

		switch (gatherWhat)
		{
		case EGT_INPUTS:
			inNames.push_back(_options[i]);
			break;
		case EGT_OUTPUTS:
			if (!core::hasFileExtension(_options[i], "baw"))
			{
				printf("Output filename must be of 'baw' extension. Ignored.\n");
				break;
			}
			outNames.push_back(_options[i]);
			break;
		default:
			printf("Ignored an input \"%s\".\n", _options[i]);
			break;
		}
	}

	if (inNames.size() != outNames.size())
	{
		printf("Fatal error. Amounts of input and output filenames doesn't match. Exiting.\n");
		return 1;
	}

	for (size_t i = 0; i < inNames.size(); ++i)
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
			continue;
		}
		if (usePwd)
			writer->writeMesh(outfile, inmesh, properties);
		else
			writer->writeMesh(outfile, inmesh, scene::EMWF_WRITE_COMPRESSED);
		outfile->drop();
	}
	writer->drop();
	device->drop();

	return 0;
}

bool checkHex(const char* _str)
{
	const size_t len = core::length(_str);

	for (size_t i = 0; i < len; ++i)
	{
		const char c = tolower(_str[i]);
		if ((c < 'a' || c > 'f') && !isdigit(c))
			return false;
	}
	return true;
}

void hexStrToIntegers(const char* _input, unsigned char* _out)
{
	for (size_t i = 0; i < 32; i += 2, ++_out)
	{
		const uint8_t val = hexCharToUint8(_input[i]) + hexCharToUint8(_input[i+1])*16;
		*_out = val;
	}
}

uint8_t hexCharToUint8(char _c)
{
	if (isdigit(_c))
		return _c - '0';
	return tolower(_c) - 'a' + 10;
}