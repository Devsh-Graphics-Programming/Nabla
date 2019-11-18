#define _IRR_STATIC_LIB_
#include <irrlicht.h>

#include "../../source/Irrlicht/COpenGLDriver.h"
//#include "../ext/ScreenShot/ScreenShot.h"
#include "../common/QToQuitEventReceiver.h"

// TODO: remove dependency
//#include "../src/irr/asset/CBAWMeshWriter.h"

using namespace irr;
using namespace core;



int main()
{
    srand(time(0));
	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check irr::SIrrlichtCreationParameters
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24; //we'd like 32bit here
	params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	params.WindowSize = dimension2d<uint32_t>(1280, 720);
	params.Fullscreen = false;
	params.Vsync = true; //! If supported by target platform
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	IrrlichtDevice* device = createDeviceEx(params);

	if (!device)
		return 1; // could not create selected driver.

	device->getCursorControl()->setVisible(false);

	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);

	video::IVideoDriver* driver = device->getVideoDriver();

	io::IFileSystem* filesystem = device->getFileSystem();
    asset::IAssetManager* am = device->getAssetManager();

    core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> ds0layout;
    {
        asset::ICPUDescriptorSetLayout::SBinding bnd[2];
        bnd[0].binding = 0u;
        bnd[0].type = asset::EDT_STORAGE_IMAGE;
        bnd[0].count = 1u;
        bnd[0].stageFlags = asset::ESS_COMPUTE;
        bnd[1] = bnd[0];
        bnd[1].binding = 1u;
        ds0layout = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(bnd, bnd+2);
    }

    core::smart_refctd_ptr<asset::ICPUComputePipeline> compPipeline;
    {
        core::smart_refctd_ptr<asset::ICPUPipelineLayout> layout;
        {
            auto ds0layout_cp = ds0layout; //do not move `ds0layout` into pipeline layout, will be needed for creation of desc set
            layout = core::make_smart_refctd_ptr<asset::ICPUPipelineLayout>(nullptr, nullptr, std::move(ds0layout_cp), nullptr, nullptr, nullptr);
        }

        core::smart_refctd_ptr<asset::ICPUSpecializedShader> cs;
        {
            auto f = core::smart_refctd_ptr<io::IReadFile>(filesystem->createAndOpenFile("../compute.comp"));

            auto cs_unspec = am->getGLSLCompiler()->createSPIRVFromGLSL(f.get(), asset::ESS_COMPUTE, "main", "comp");
            auto specInfo = core::make_smart_refctd_ptr<asset::ISpecializationInfo>(core::vector<asset::SSpecializationMapEntry>{}, nullptr, "main", asset::ESS_COMPUTE);

            cs = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(cs_unspec), std::move(specInfo));
        }

        compPipeline = core::make_smart_refctd_ptr<asset::ICPUComputePipeline>(nullptr, std::move(layout), std::move(cs));
    }

    asset::IAssetLoader::SAssetLoadParams lparams;
    auto loaded = am->getAsset("../img.png", lparams);
    auto inImg = core::smart_refctd_ptr<asset::ICPUImage>(static_cast<asset::ICPUImage*>(loaded.getContents().first->get()));
    core::smart_refctd_ptr<asset::ICPUImage> outImg;
    {
        asset::ICPUImage::SCreationParams imgInfo;
        imgInfo.arrayLayers = 1u;
        imgInfo.extent = inImg->getCreationParameters().extent;
        imgInfo.flags = static_cast<asset::ICPUImage::E_CREATE_FLAGS>(0u);
        imgInfo.format = inImg->getCreationParameters().format;
        imgInfo.mipLevels = 1u;
        imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
        imgInfo.type = asset::ICPUImage::ET_2D;

        outImg = asset::ICPUImage::create(std::move(imgInfo));
    }

    core::smart_refctd_ptr<asset::ICPUImageView> inImgView;
    core::smart_refctd_ptr<asset::ICPUImageView> outImgView;
    {
        asset::ICPUImageView::SCreationParams info1;
        info1.flags = static_cast<asset::ICPUImageView::E_CREATE_FLAGS>(0u);
        info1.format = inImg->getCreationParameters().format;
        info1.image = inImg;
        info1.viewType = asset::ICPUImageView::ET_2D;
        asset::IImage::SSubresourceRange subresRange;
        subresRange.baseArrayLayer = 0u;
        subresRange.layerCount = 1u;
        subresRange.baseMipLevel = 0u;
        subresRange.levelCount = 1u;
        info1.subresourceRange = subresRange;

        asset::ICPUImageView::SCreationParams info2 = info1;
        info2.image = outImg;

        inImgView = asset::ICPUImageView::create(std::move(info1));
        outImgView = asset::ICPUImageView::create(std::move(info2));
    }

    auto ds0 = core::make_smart_refctd_ptr<asset::ICPUDescriptorSet>(std::move(ds0layout));
    {
        asset::ICPUDescriptorSet::SDescriptorInfo info[2];
        info[0].image.imageLayout = asset::EIL_UNDEFINED;
        info[0].image.sampler = nullptr;
        info[1].assign(info[0], asset::EDT_STORAGE_IMAGE);
        info[1].desc = std::move(outImgView);
        info[0].desc = info[1].desc; //std::move(inImgView); //TODO
        asset::ICPUDescriptorSet::SWriteDescriptorSet writes[2];
        writes[0].arrayElement = 0u;
        writes[0].binding = 0u;
        writes[0].count = 1u;
        writes[0].descriptorType = asset::EDT_STORAGE_IMAGE;
        writes[0].info = info;
        writes[1] = writes[0];
        writes[1].binding = 1u;
        writes[1].info = info+1;
        ds0->updateDescriptorSet(2u, writes, 0u, nullptr);
    }

    core::smart_refctd_ptr<video::IGPUDescriptorSet> ds0_gpu;
    {
        asset::ICPUDescriptorSet* ds0_rawptr = ds0.get();
        ds0_gpu = driver->getGPUObjectsFromAssets(&ds0_rawptr, (&ds0_rawptr)+1)->front();
    }
    core::smart_refctd_ptr<video::IGPUComputePipeline> compPipeline_gpu;
    {
        asset::ICPUComputePipeline* cp_rawptr = compPipeline.get();
        compPipeline_gpu = driver->getGPUObjectsFromAssets(&cp_rawptr, (&cp_rawptr)+1)->front();
    }

	return 0;
}
