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
		void presentImageOnScreen(core::smart_refctd_ptr<video::IGPUImageView>&& inTex, core::smart_refctd_ptr<video::IGPUImageView>&& outTex=nullptr)
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
            IGPUImage* gpuimg = nullptr;
			video::IDriverMemoryBacked::SDriverMemoryRequirements reqs;
			IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<ViewOrImage,IGPUImageView>::value)
			{
				reqs.vulkanReqs.size = tex->getCreationParameters().image->getImageDataSizeInBytes();
                gpuimg = tex->getCreationParameters().image.get();
			}
			IRR_PSEUDO_ELSE_CONSTEXPR
			{
				reqs.vulkanReqs.size = tex->getImageDataSizeInBytes();
                gpuimg = tex;
			}
			IRR_PSEUDO_IF_CONSTEXPR_END
			reqs.vulkanReqs.alignment = 64u;
			reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
			reqs.memoryHeapLocation = video::IDriverMemoryAllocation::ESMT_NOT_DEVICE_LOCAL;
			reqs.mappingCapability = video::IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_READ|video::IDriverMemoryAllocation::EMCF_COHERENT|video::IDriverMemoryAllocation::EMCF_CACHED;
			auto buffer = driver->createGPUBufferOnDedMem(reqs);

			auto fence = ext::ScreenShot::createScreenShot(driver,gpuimg,buffer.get());
			while (fence->waitCPU(1000ull,fence->canDeferredFlush())==video::EDFR_TIMEOUT_EXPIRED) {}

			auto alloc = buffer->getBoundMemory();
			alloc->mapMemoryRange(video::IDriverMemoryAllocation::EMCAF_READ,{0u,reqs.vulkanReqs.size});
			auto cpubuffer = core::make_smart_refctd_ptr<asset::CCustomAllocatorCPUBuffer<core::null_allocator<uint8_t> > >(reqs.vulkanReqs.size,alloc->getMappedPointer(),core::adopt_memory);

            auto calcPitchInBlocks = [](uint32_t width, uint32_t blockByteSize) -> uint32_t
            {
                auto rowByteSize = width * blockByteSize;
                for (uint32_t _alignment = 8u; _alignment > 1u; _alignment >>= 1u)
                {
                    auto paddedSize = core::alignUp(rowByteSize, _alignment);
                    if (paddedSize % blockByteSize)
                        continue;
                    return paddedSize / blockByteSize;
                }
                return width;
            };
            auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1u);
            ICPUImage::SBufferCopy& region = regions->front();
            region.imageSubresource.mipLevel = 0u;
            region.imageSubresource.baseArrayLayer = 0u;
            region.imageSubresource.layerCount = 1u;
            region.bufferOffset = 0u;
            region.bufferRowLength = calcPitchInBlocks(gpuimg->getCreationParameters().extent.width, getTexelOrBlockBytesize(gpuimg->getCreationParameters().format));
            region.bufferImageHeight = 0u; //tightly packed
            region.imageOffset = { 0u, 0u, 0u };
            region.imageExtent = gpuimg->getCreationParameters().extent;
			
			auto assMgr = device->getAssetManager();
			IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<ViewOrImage,IGPUImageView>::value)
			{
				const auto& origViewParams = tex->getCreationParameters();

				asset::ICPUImage::SCreationParams params = origViewParams.image->getCreationParameters();
				auto img = asset::ICPUImage::create(std::move(params));
                img->setBufferAndRegions(std::move(cpubuffer), regions);

                asset::ICPUImageView::SCreationParams viewParams;
                viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(origViewParams.flags);
                viewParams.image = std::move(img);
                viewParams.viewType = static_cast<IImageView<ICPUImage>::E_TYPE>(origViewParams.viewType);
                viewParams.format = origViewParams.format;
                memcpy(&viewParams.components, &origViewParams.components, sizeof(viewParams.components));
                memcpy(&viewParams.subresourceRange, &origViewParams.subresourceRange, sizeof(viewParams.subresourceRange));
				auto imgView = core::make_smart_refctd_ptr<asset::ICPUImageView>(std::move(viewParams));
				assMgr->writeAsset(outname,asset::IAssetWriter::SAssetWriteParams(imgView.get()));
			}
			IRR_PSEUDO_ELSE_CONSTEXPR //idk if this 'else' should even exist, maybe it should always write ICPUImageView
			{
				asset::ICPUImage::SCreationParams params = tex->getCreationParameters();
				auto img = asset::ICPUImage::create(std::move(params));
                img->setBufferAndRegions(std::move(cpubuffer), regions);

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

			IGPUDescriptorSetLayout::SBinding binding{0u,EDT_COMBINED_IMAGE_SAMPLER,1u,IGPUSpecializedShader::ESS_FRAGMENT,nullptr};
			dsLayout = driver->createGPUDescriptorSetLayout(&binding,&binding+1u);

			{
				auto pLayout = driver->createGPUPipelineLayout(nullptr,nullptr,core::smart_refctd_ptr(dsLayout),nullptr,nullptr,nullptr);

                IAssetLoader::SAssetLoadParams lp;
				auto fs_bundle = device->getAssetManager()->getAsset("../present.frag", lp);
				auto fs_contents = fs_bundle.getContents();
				if (fs_contents.first==fs_contents.second)
					return false;
                ICPUSpecializedShader* fs = static_cast<ICPUSpecializedShader*>(fs_contents.first->get());

				auto fragShader = driver->getGPUObjectsFromAssets(&fs, &fs+1)->front();
                if (!fragShader)
                    return false;

				IGPUSpecializedShader* shaders[2] = {std::get<0>(fullScreenTriangle).get(),fragShader.get()};
				SBlendParams blendParams;
				blendParams.logicOpEnable = false;
				blendParams.logicOp = ELO_NO_OP;
				for (size_t i=0ull; i<SBlendParams::MAX_COLOR_ATTACHMENT_COUNT; i++)
                    blendParams.blendParams[i].attachmentEnabled = (i==0ull);
                SRasterizationParams rasterParams;
                rasterParams.faceCullingMode = EFCM_NONE;
                rasterParams.depthCompareOp = ECO_ALWAYS;
                rasterParams.minSampleShading = 1.f;
                rasterParams.depthWriteEnable = false;
                rasterParams.depthTestEnable = false;

				presentPipeline = driver->createGPURenderpassIndependentPipeline(	nullptr,std::move(pLayout),shaders,shaders+sizeof(shaders)/sizeof(IGPUSpecializedShader*),
																					std::get<SVertexInputParams>(fullScreenTriangle),blendParams,
																					std::get<SPrimitiveAssemblyParams>(fullScreenTriangle),rasterParams);
			}

			{
                SBufferBinding<IGPUBuffer> idxBinding{0ull,nullptr};
				presentMB = core::make_smart_refctd_ptr<IGPUMeshBuffer>(nullptr, nullptr, nullptr, std::move(idxBinding));
				presentMB->setIndexCount(3u);
				presentMB->setInstanceCount(1u);
			}

			return true;
		}
		
		void testImage(const std::string& path)
		{
			os::Printer::log("Reading", path);
	
			auto* assetMgr = device->getAssetManager();

            core::smart_refctd_ptr<ICPUImageView> actualcputex;
			auto cputex = assetMgr->getAsset(path, {});
			auto contents = cputex.getContents();
			if (contents.first!=contents.second)
			{
				io::path filename, extension;
				core::splitFilename(path.c_str(), nullptr, &filename, &extension);
				filename += "."; filename += extension;
		
                //TODO @anastazluk decide if this is needed at all or should be redefined somehow
				bool writeable = (extension != "dds");
		
				auto asset = *contents.first;
				core::smart_refctd_ptr<video::IGPUImageView> imgView;
				switch (asset->getAssetType())
				{
					case IAsset::ET_IMAGE:
                        {
                            ICPUImageView::SCreationParams viewParams;
                            viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
                            viewParams.image = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(asset);
                            viewParams.format = viewParams.image->getCreationParameters().format;
                            viewParams.viewType = IImageView<ICPUImage>::ET_2D;
                            viewParams.subresourceRange.baseArrayLayer = 0u;
                            viewParams.subresourceRange.layerCount = 1u;
                            viewParams.subresourceRange.baseMipLevel = 0u;
                            viewParams.subresourceRange.levelCount = 1u;

                            actualcputex = ICPUImageView::create(std::move(viewParams));
                            imgView = driver->getGPUObjectsFromAssets(&actualcputex.get(), &actualcputex.get()+1u)->front();
                        }
						break;
					case IAsset::ET_IMAGE_VIEW:
						{
							actualcputex = core::smart_refctd_ptr_static_cast<asset::ICPUImageView>(asset);
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
					viewParams.image = driver->createDeviceLocalGPUImageOnDedMem(video::IGPUImage::SCreationParams(viewParams.image->getCreationParameters()));
					viewParams.viewType = IGPUImageView::ET_2D;
					auto outView = driver->createGPUImageView(std::move(viewParams));
					presentImageOnScreen(core::smart_refctd_ptr(imgView),core::smart_refctd_ptr(outView));
					if (writeable)
						dumpTextureToFile(outView.get(), (io::path("screen_") + filename).c_str());
				}
		
				if (writeable)
				{
					asset::IAssetWriter::SAssetWriteParams wparams(actualcputex.get());

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

