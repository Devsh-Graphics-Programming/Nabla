#define _IRR_STATIC_LIB_
#include <irrlicht.h>
#include <iostream>
#include <cstdio>
#include <fstream>

#include "../../ext/FullScreenTriangle/FullScreenTriangle.h"
#include "../../ext/ScreenShot/ScreenShot.h"


using namespace irr;
using namespace core;
using namespace asset;
using namespace video;

class App
{
		void presentImageOnScreen(core::smart_refctd_ptr<video::ITexture>&& inTex, core::smart_refctd_ptr<video::ITexture>&& outTex=nullptr)
		{
			IFrameBuffer* framebuffer = nullptr;
			if (outTex)
			{
				framebuffer = driver->addFrameBuffer();
				framebuffer->attach(video::EFAP_COLOR_ATTACHMENT0, std::move(outTex), 0u);
			}

			auto ds = driver->createGPUDescriptorSet(core::smart_refctd_ptr(dsLayout));
			IGPUDescriptorSet::SDescriptorInfo info;
			{
				info.desc = std::move(inTex);
				ISampler::SParams samplerParams = {ISampler::ETC_CLAMP_TO_EDGE,ISampler::ETC_CLAMP_TO_EDGE,ISampler::ETC_CLAMP_TO_EDGE,ISampler::ETBC_FLOAT_OPAQUE_BLACK,ISampler::ETF_LINEAR,ISampler::ETF_LINEAR,ISampler::ESMM_LINEAR,0u,false,ECO_ALWAYS};
				info.image = {driver->createGPUSampler(samplerParams),EIL_SHADER_READ_ONLY_OPTIMAL};
			}
			IGPUDescriptorSet::SWriteDescriptorSet write = {ds.get(),0u,0u,1u,EDT_COMBINED_IMAGE_SAMPLER,&info};
			driver->updateDescriptorSets(1u,&write,0u,nullptr);

			device->run();
			driver->beginScene(true,true);
			// so we have something to download
			driver->bindGraphicsPipeline(presentPipeline.get());
			driver->bindDescriptorSets(EPBP_GRAPHICS,presentPipeline->getLayout(),0u,1u,&ds.get(),nullptr);
			if (outTex)
			{
				driver->setRenderTarget(framebuffer);
				driver->drawMeshBuffer(presentMB.get()); 
				driver->setRenderTarget(nullptr,false);
			}
			//
			driver->drawMeshBuffer(presentMB.get());
			//
			driver->endScene();

			driver->removeFrameBuffer(framebuffer);
		}
		template<class ViewOrImage>
		void dumpTextureToFile(ViewOrImage* tex, const std::string& outname)
		{
			video::IDriverMemoryBacked::SDriverMemoryRequirements reqs;
			IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<ViewOrImage,IGPUImageView>::value)
			{
				reqs.vulkanReqs.size = tex->getCreationParams().image->getImageDataSizeInBytes();
			}
			IRR_PSEUDO_ELSE_CONSTEXPR
			{
				reqs.vulkanReqs.size = tex->getImageDataSizeInBytes();
			}
			IRR_PSEUDO_IF_CONSTEXPR_END
			reqs.vulkanReqs.alignment = 64u;
			reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
			reqs.memoryHeapLocation = video::IDriverMemoryAllocation::ESMT_NOT_DEVICE_LOCAL;
			reqs.mappingCapability = video::IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_READ|video::IDriverMemoryAllocation::EMCF_COHERENT|video::IDriverMemoryAllocation::EMCF_CACHED;
			auto buffer = driver->createGPUBufferOnDedMem(reqs);

			auto fence = ext::ScreenShot::createScreenShot(driver,tex,buffer);
			while (fence->waitCPU(1000ull,fence->canDeferredFlush())==video::EDFR_TIMEOUT_EXPIRED) {}

			auto alloc = buffer->getBoundMemory();
			alloc->mapMemoryRange(video::IDriverMemoryAllocation::EMCAF_READ,{0u,reqs.vulkanReqs.size});
			auto cpubuffer = core::make_smart_refctd_ptr<asset::CCustomAllocatorCPUBuffer<core::null_allocator<typename Allocator::value_type> > >(reqs.vulkanReqs.size,alloc->getMappedPointer(),core::adopt_memory);
			
			auto assMgr = device->getAssetManager();
			IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<ViewOrImage,IGPUImageView>::value)
			{
				const auto& origViewParams = tex->getCreationParams();

				asset::ICPUImage::SCreationParams params = origViewParams.image->getCreationParams();
				auto img = asset::ICPUImage::create(std::move(params));

				asset::IImageView::SCreationParams viewParams = {origViewParams.flags,std::move(img),origViewParams.viewType,origViewParams.format,origViewParams.components,origViewParams.subresourceRange};
				auto imgView = core::make_smart_refctd_ptr<asset::ICPUImageView>(std::move(viewParams));
				assMgr->writeAsset(outname,asset::IAssetWriter::SAssetWriteParams(imgView.get()));
			}
			IRR_PSEUDO_ELSE_CONSTEXPR
			{
				asset::ICPUImage::SCreationParams params = tex->getCreationParams();
				auto img = asset::ICPUImage::create(std::move(params));
				assMgr->writeAsset(outname,asset::IAssetWriter::SAssetWriteParams(img.get()));
			}
			IRR_PSEUDO_IF_CONSTEXPR_END
		}

		core::smart_refctd_ptr<IrrlichtDevice> device;
		IVideoDriver* driver;

		core::smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout;
		core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> presentPipeline;
		core::smart_refctd_ptr<IGPUMeshBuffer> presentMB;
	public:
		App() = default;

		bool init()
		{
			irr::SIrrlichtCreationParameters params;
			params.Bits = 24; //may have to set to 32bit for some platforms
			params.ZBufferBits = 24; //we'd like 32bit here
			params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
			params.WindowSize = dimension2d<uint32_t>(1600, 900);
			params.Fullscreen = false;

			device = createDeviceEx(params);
			if (!device)
				return false;

			driver = device->getVideoDriver();

			auto fullScreenTriangle = ext::FullScreenTriangle::createFullScreenTriangle(driver);

			IGPUDescriptorSetLayout::SBinding binding{0u,EDT_COMBINED_IMAGE_SAMPLER,1u,ESS_FRAGMENT,nullptr};
			dsLayout = driver->createGPUDescriptorSetLayout(&binding,&binding+1u);

			{
				auto pLayout = driver->createGPUPipelineLayout(nullptr,nullptr,core::smart_refctd_ptr(dsLayout),nullptr,nullptr,nullptr);

				auto shaderSource = device->getAssetManager()->getAsset("../present.frag", {});
				auto contents = shaderSource.getContents();
				if (contents.first==contents.second)
					return false;

				auto unspecShader = driver->createGPUShader(core::smart_refctd_ptr_static_cast<ICPUShader>(*contents.first));
				if (!unspecShader)
					return false;
				auto fragShader = driver->createGPUSpecializedShader(unspecShader.get(),nullptr);

				IGPUSpecializedShader* shaders[2] = {std::get<0>(fullScreenTriangle).get(),fragShader.get()};
				SBlendParams blendParams;
				blendParams.logicOpEnable = false;
				blendParams.logicOp = ELO_NO_OP;
				for (size_t i=0ull; i<SBlendParams::MAX_COLOR_ATTACHMENT_COUNT; i++)
					blendParams.blendParams[i] = {i==0ull,false,EBF_ONE,EBF_ZERO,EBO_ADD,EBF_ONE,EBF_ZERO,EBO_ADD,0xfu};
				SStencilOpParams defaultStencil;
				SRasterizationParams rasterParams = {1u,EPM_FILL,EFCM_NONE,ECO_ALWAYS,IImage::ESCF_1_BIT,{~0u,~0u},1.f,0.f,0.f,defaultStencil,defaultStencil,
														{false,false,true,false,false,false,false,false,false,false,false}};

				presentPipeline = driver->createGPURenderpassIndependentPipeline(	nullptr,std::move(pLayout),shaders,shaders+sizeof(shaders)/sizeof(IGPUSpecializedShader*),
																					std::get<SVertexInputParams>(fullScreenTriangle),blendParams,
																					std::get<SPrimitiveAssemblyParams>(fullScreenTriangle),rasterParams);
			}

			{
				presentMB = core::make_smart_refctd_ptr<IGPUMeshBuffer>();
				presentMB->setIndexCount(3u);
				presentMB->setInstanceCount(1u);
			}

			return true;
		}
		
		void testImage(const std::string& path)
		{
			os::Printer::log("Reading", path);
	
			auto* assetMgr = device->getAssetManager();

			auto cputex = assetMgr->getAsset(path, {});
			auto contents = cputex.getContents();
			if (contents.first!=contents.second)
			{
				io::path filename, extension;
				core::splitFilename(path.c_str(), nullptr, &filename, &extension);
				filename += "."; filename += extension;
		
				bool writeable = (extension != "dds");
		
				auto asset = *contents.first;
				core::smart_refctd_ptr<video::IGPUImageView> imgView;
				switch (asset->getAssetType())
				{
					case IAsset::ET_IMAGE:
						break;
					case IAsset::ET_IMAGE_VIEW:
						{
							auto actualcputex = core::smart_refctd_ptr_static_cast<asset::ICPUImageView>(asset);
							imgView = driver->getGPUObjectsFromAssets(&actualcputex.get(),&actualcputex.get()+1u)->front();
						}
						break;
					default:
						assert(false);
						break;
				}
				if (imgView)
				{
					auto viewParams = imgView->getCreationParameters();
					viewParams.image = driver->createGPUImage(video::IGPUImage::SCreationParams(viewParams.image->getCreationParameters()));
					viewParams.viewType = IGPUImageView::ET_2D;
					auto outView = driver->createGPUImageView(std::move(viewParams));
					presentImageOnScreen(core::smart_refctd_ptr(imgView),core::smart_refctd_ptr(outView));
					if (writeable)
						dumpTextureToFile(outView.get(), (io::path("screen_") + filename).c_str());
				}
		
				if (writeable)
				{
					asset::CImageData* img = actualcputex->getMipMap(0u).first[0];
					asset::IAssetWriter::SAssetWriteParams wparams(img);

					assetMgr->writeAsset((io::path("write_")+filename).c_str(), wparams);
				}
				assetMgr->removeAssetFromCache(cputex);
				//assetMgr->removeCachedGPUObject(,tex); // TODO: provide a variant of `removeCachedGPUObject` that does not leak (as in, removes all children too)
			}
			else
				std::cout << "ERROR: CANNOT LOAD FILE!" << std::endl;
		}
};

int main()
{	
	App app;
	if (!app.init())
		return 1;

	std::ifstream list("../testlist.txt");
	if (list.is_open())
	{
        std::string line;
        for (; std::getline(list, line); )
        {
			if(line != "" && line[0] != ';')
				app.testImage(line);
        }
	}

	return 0;
}

