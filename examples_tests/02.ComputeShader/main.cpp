#define _IRR_STATIC_LIB_
#include <irrlicht.h>

#include "../../source/Irrlicht/COpenGLDriver.h"
//#include "../ext/ScreenShot/ScreenShot.h"
#include "../common/QToQuitEventReceiver.h"


using namespace irr;
using namespace core;


int main()
{
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

	asset::IAsset* ds0layoutHandle = nullptr;
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
		ds0layoutHandle = ds0layout.get();
    }

    core::smart_refctd_ptr<video::IGPUDescriptorSet> ds0_gpu;
	asset::ICPUImageView* outImgViewRawPtr;
	uint32_t imgSize[2];
    {
		asset::IAssetLoader::SAssetLoadParams lparams;
		auto loaded = am->getAsset("../../media/color_space_test/R8G8B8A8_2.png", lparams);
		auto inImg = core::smart_refctd_ptr<asset::ICPUImage>(static_cast<asset::ICPUImage*>(loaded.getContents().first->get()));
		core::smart_refctd_ptr<asset::ICPUImage> outImg;
		{
			asset::ICPUImage::SCreationParams imgInfo = inImg->getCreationParameters();
			imgInfo.flags = static_cast<asset::ICPUImage::E_CREATE_FLAGS>(0u);
			imgSize[0] = imgInfo.extent.width;
			imgSize[1] = imgInfo.extent.height;

			outImg = asset::ICPUImage::create(std::move(imgInfo));
		}

		core::smart_refctd_ptr<asset::ICPUImageView> inImgView;
		core::smart_refctd_ptr<asset::ICPUImageView> outImgView;
		{
			asset::ICPUImageView::SCreationParams info1;
			info1.flags = static_cast<asset::ICPUImageView::E_CREATE_FLAGS>(0u);
			// storage images do not support SRGB views
			assert(inImg->getCreationParameters().format==asset::EF_R8G8B8A8_SRGB);
			info1.format = asset::EF_R8G8B8A8_UNORM;
			info1.image = inImg;
			info1.viewType = asset::ICPUImageView::ET_2D;
			asset::IImage::SSubresourceRange& subresRange = info1.subresourceRange;
			subresRange.baseArrayLayer = 0u;
			subresRange.layerCount = 1u;
			subresRange.baseMipLevel = 0u;
			subresRange.levelCount = 1u;

			asset::ICPUImageView::SCreationParams info2 = info1;
			info2.image = outImg;

			inImgView = asset::ICPUImageView::create(std::move(info1));
			outImgView = asset::ICPUImageView::create(std::move(info2));
			outImgViewRawPtr = outImgView.get();
		}

		auto ds0 = core::make_smart_refctd_ptr<asset::ICPUDescriptorSet>(std::move(ds0layout));
		{
			auto descriptor = ds0->getDescriptors(0).begin();
			descriptor->desc = std::move(inImgView);
			descriptor->image.imageLayout = asset::EIL_UNDEFINED;
			descriptor->image.sampler = nullptr;
			descriptor = ds0->getDescriptors(1).begin();
			descriptor->desc = std::move(outImgView);
			descriptor->image.imageLayout = asset::EIL_UNDEFINED;
			descriptor->image.sampler = nullptr;
		}

        asset::ICPUDescriptorSet* ds0_rawptr = ds0.get();
        ds0_gpu = driver->getGPUObjectsFromAssets(&ds0_rawptr, (&ds0_rawptr)+1)->front();
    }
	
	core::smart_refctd_ptr<video::IGPUComputePipeline> compPipeline;
    {
        core::smart_refctd_ptr<video::IGPUPipelineLayout> layout;
        {
			asset::SPushConstantRange range;
			range.offset = 0u;
			range.size = sizeof(uint32_t)*2u;
			range.stageFlags = asset::ESS_COMPUTE;
            layout = driver->createGPUPipelineLayout(&range,&range+1,core::smart_refctd_ptr_dynamic_cast<video::IGPUDescriptorSetLayout>(am->findGPUObject(ds0layoutHandle)),nullptr,nullptr,nullptr);
        }
		core::smart_refctd_ptr<video::IGPUSpecializedShader> shader;
        {
            auto f = core::smart_refctd_ptr<io::IReadFile>(filesystem->createAndOpenFile("../compute.comp"));

            auto cs_unspec = am->getGLSLCompiler()->createSPIRVFromGLSL(f.get(), asset::ESS_COMPUTE, "main", "comp");
            auto specInfo = core::make_smart_refctd_ptr<asset::ISpecializationInfo>(core::vector<asset::SSpecializationMapEntry>{}, nullptr, "main", asset::ESS_COMPUTE);

            auto cs = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(cs_unspec), std::move(specInfo));
			auto cs_rawptr = cs.get();
			shader = driver->getGPUObjectsFromAssets(&cs_rawptr,&cs_rawptr+1)->front();
        }

		compPipeline = driver->createGPUComputePipeline(nullptr, std::move(layout), std::move(shader));
    }

	auto blitFBO = driver->addFrameBuffer();
	blitFBO->attach(video::EFAP_COLOR_ATTACHMENT0,core::smart_refctd_ptr_dynamic_cast<video::IGPUImageView>(am->findGPUObject(outImgViewRawPtr)));

	while (device->run())
	{
		driver->beginScene(true);

		driver->bindComputePipeline(compPipeline.get());
		const video::IGPUDescriptorSet* descriptorSet = ds0_gpu.get();
		driver->bindDescriptorSets(video::EPBP_COMPUTE, compPipeline->getLayout(),0u,1u,&descriptorSet,nullptr);
		driver->pushConstants(compPipeline->getLayout(),asset::ESS_COMPUTE,0u,sizeof(imgSize),imgSize);
		driver->dispatch((imgSize[0]+15u)/16u,(imgSize[1]+15u)/16u,1u);

		video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT|GL_TEXTURE_FETCH_BARRIER_BIT);
		driver->blitRenderTargets(blitFBO,nullptr,false,false);

		driver->endScene();
	}

	device->drop();
	return 0;
}
