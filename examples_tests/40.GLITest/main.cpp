#define _IRR_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <irrlicht.h>

using namespace irr;
using namespace asset;
using namespace core;

void loadAndWriteImageForTesting(const std::string_view& path, IAssetManager* assetManager)
{
	auto fileName = std::string(io::IFileSystem::getFileBasename(path.data()).c_str());

	asset::IAssetLoader::SAssetLoadParams loadingParams;
	auto image_bundle = assetManager->getAsset(path.data(), loadingParams);
	if (!image_bundle.isEmpty())
	{
		auto image = image_bundle.getContents().first[0];

		IAssetWriter::SAssetWriteParams writeParams(image.get());
		assetManager->writeAsset(fileName, writeParams);
	}
}

constexpr std::initializer_list<std::string_view> gliTestingPaths =
{
	/*
		.dds files - those we will be testing, 
		uncomment for testing purposes
	*/

	"../../media/GLI/array_r8_uint.dds",																					
	"../../media/GLI/cube_rgba8_unorm.dds",		
	"../../media/GLI/kueken7_a8_unorm.dds",			// it is almost written by your side, but doesn't create by GLI API side - not sure if you write to GLI texture correctly, maybe final size is wrong                >	imagemanipulatortest_d.exe!heap_alloc_dbg_internal(const unsigned __int64 size, const int block_use, const char * const file_name, const int line_number) Line 359	C++
	"../../media/GLI/kueken7_rgba_dxt1_srgb.dds",		// curious - it has some mipmaps, but passed through your writing and didn't save by GLI API side, maybe you write to GLI texture layout incorrectly				>	imagemanipulatortest_d.exe!heap_alloc_dbg_internal(const unsigned __int64 size, const int block_use, const char * const file_name, const int line_number) Line 359	C++
	"../../media/GLI/kueken7_rgba_dxt3_unorm.dds",		// but that one with diferrent BC fails on copy, maybe because of your block offsets																				>	imagemanipulatortest_d.exe!irr::asset::CGLIWriter::writeGLIFile::__l2::<lambda>(unsigned int ptrOffset, const irr::core::vectorSIMD_32<unsigned int> & texelCoord) Line 170	C++
	"../../media/GLI/kueken7_rgba_dxt5_srgb.dds",		// it also passed, but could't be created by GLI API side																											>	imagemanipulatortest_d.exe!heap_alloc_dbg_internal(const unsigned __int64 size, const int block_use, const char * const file_name, const int line_number) Line 359	C++
	"../../media/GLI/earth-cubemap.dds",				// fails immediately on texels copy																																	> 	imagemanipulatortest_d.exe!irr::asset::CGLIWriter::writeGLIFile::__l2::<lambda>(unsigned int ptrOffset, const irr::core::vectorSIMD_32<unsigned int> & texelCoord) Line 170	C++
	"../../media/GLI/earth-cubemap2.dds"				// fails immediately on texels copy																																	>  	imagemanipulatortest_d.exe!irr::asset::CGLIWriter::writeGLIFile::__l2::<lambda>(unsigned int ptrOffset, const irr::core::vectorSIMD_32<unsigned int> & texelCoord) Line 170	C++

	/*
		.ktx files
		dont touch it right now
	*/

	//"../../media/GLI/texturearray_astc_8x8_unorm.ktx", 
	//"../../media/GLI/texturearray_bc3_unorm.ktx", 
	//"../../media/GLI/texturearray_etc2_unorm.ktx" 
};

int main()
{
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24; 
	params.ZBufferBits = 24; 
	params.DriverType = video::EDT_OPENGL; 
	params.WindowSize = dimension2d<uint32_t>(1280, 720);
	params.Fullscreen = false;
	params.Vsync = true;
	params.Doublebuffer = true;
	params.Stencilbuffer = false; 
	auto device = createDeviceEx(params);

	if (!device)
		return 1;

	auto driver = device->getVideoDriver();
	auto assetManager = device->getAssetManager();

	for (const auto& gliPath : gliTestingPaths)
		loadAndWriteImageForTesting(gliPath, assetManager);
}
