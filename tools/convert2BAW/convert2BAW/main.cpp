#define _IRR_STATIC_LIB_

#include <irrlicht.h>
#include <vector>

using namespace irr;

enum E_GATHER_TARGET
{
	EGT_UNDEFINED = 0,
	EGT_INPUTS,
	EGT_OUTPUTS
};

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
	scene::IMeshWriter* const writer = smgr->createMeshWriter(irr::scene::EMWT_BAW);

	std::vector<const char*> inNames;
	std::vector<const char*> outNames;

	E_GATHER_TARGET gatherWhat = EGT_UNDEFINED;
	for (size_t i = 0; i < _optCnt; ++i)
	{
		if (_options[i][0] == '-')
		{
			if (core::equalsIgnoreCase("i", _options[i]+1))
				gatherWhat = EGT_INPUTS;
			else if (core::equalsIgnoreCase("o", _options[i]+1))
				gatherWhat = EGT_OUTPUTS;
			else
			{
				gatherWhat = EGT_UNDEFINED;
				printf("Ignored unrecognized option \"%s\".\n", _options[i]);
			}
			continue;
		}

		switch (gatherWhat)
		{
		case 1:
			inNames.push_back(_options[i]);
			break;
		case 2:
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
			printf("Could not load mesh %s.", inNames[i]);
			continue;
		}
		io::IWriteFile* outfile = fs->createAndWriteFile(outNames[i]);
		if (!outfile)
		{
			printf("Could not create/open file %s.", outNames[i]);
			continue;
		}
		writer->writeMesh(outfile, inmesh);
		outfile->drop();
	}
	device->drop();

	return 0;
}