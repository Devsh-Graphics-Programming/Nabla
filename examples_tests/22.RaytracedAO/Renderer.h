#ifndef _RENDERER_INCLUDED_
#define _RENDERER_INCLUDED_

#include "nabla.h"
#include "nbl/ext/RadeonRays/RadeonRays.h"
// pesky leaking defines
#undef PI

#include "nbl/ext/MitsubaLoader/CMitsubaLoader.h"

#include <ISceneManager.h>

#ifdef _NBL_BUILD_OPTIX_
#include "nbl/ext/OptiX/Manager.h"
#endif

#include <thread>
#include <future>

class Renderer : public nbl::core::IReferenceCounted, public nbl::core::InterfaceUnmovable
{
    public:
		#include "rasterizationCommon.h"
		#include "raytraceCommon.h"
		#ifdef __cplusplus
			#undef uint
			#undef vec4
			#undef mat4
			#undef mat4x3
		#endif

		Renderer(nbl::video::IVideoDriver* _driver, nbl::asset::IAssetManager* _assetManager, nbl::scene::ISceneManager* _smgr, bool useDenoiser = true);

		void initSceneResources(nbl::asset::SAssetBundle& meshes);

		void deinitSceneResources();
		
		void initScreenSizedResources(uint32_t width, uint32_t height, nbl::core::smart_refctd_ptr<nbl::asset::ICPUBuffer>&& sampleSequence);

		void deinitScreenSizedResources();

		void resetSampleAndFrameCounters();

		void takeAndSaveScreenShot(const std::filesystem::path& screenshotFilePath);

		bool render(nbl::ITimer* timer);

		auto* getColorBuffer() { return m_colorBuffer; }

		const auto& getSceneBound() const { return m_sceneBound; }
		
		uint64_t getSamplesPerPixelPerDispatch() const
		{
			return m_staticViewData.samplesPerPixelPerDispatch;
		}
		uint64_t getTotalSamplesPerPixelComputed() const
		{
			const auto framesDispatched = static_cast<uint64_t>(m_framesDispatched);
			return framesDispatched*m_staticViewData.samplesPerPixelPerDispatch;
		}
		uint64_t getTotalSamplesComputed() const
		{
			const auto samplesPerDispatch = static_cast<uint64_t>(m_staticViewData.samplesPerPixelPerDispatch*m_staticViewData.imageDimensions.x*m_staticViewData.imageDimensions.y);
			const auto framesDispatched = static_cast<uint64_t>(m_framesDispatched);
			return framesDispatched*samplesPerDispatch;
		}
		uint64_t getTotalRaysCast() const
		{
			return m_totalRaysCast;
		}

		//! Brief guideline to good path depth limits
		// Want to see stuff with indirect lighting on the other side of a pane of glass
		// 5 = glass frontface->glass backface->diffuse surface->diffuse surface->light
		// Want to see through a glass box, vase, or office 
		// 7 = glass frontface->glass backface->glass frontface->glass backface->diffuse surface->diffuse surface->light
		// pick higher numbers for better GI and less bias
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t MaxPathDepth = 8u;
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t RandomDimsPerPathVertex = 3u;
		// one less because the first path vertex is rasterized
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t MaxDimensions = RandomDimsPerPathVertex*(MaxPathDepth-1u);

		//
		static constexpr inline uint32_t AntiAliasingSequenceLength = 1024;
		static const float AntiAliasingSequence[AntiAliasingSequenceLength][2];
    protected:
        ~Renderer();

		struct InitializationData
		{
			InitializationData() : lights(),lightCDF() {}
			InitializationData(InitializationData&& other) : InitializationData()
			{
				operator=(std::move(other));
			}
			~InitializationData() {lightCDF.~vector(); }

			inline InitializationData& operator=(InitializationData&& other)
			{
				lights = std::move(other.lights);
				lightCDF = std::move(other.lightCDF);
				return *this;
			}

			nbl::core::vector<SLight> lights;
			union
			{
				nbl::core::vector<float> lightPDF;
				nbl::core::vector<uint32_t> lightCDF;
			};
		};
		InitializationData initSceneObjects(const nbl::asset::SAssetBundle& meshes);
		void initSceneNonAreaLights(InitializationData& initData);
		void finalizeScene(InitializationData& initData);

		//
		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> createScreenSizedTexture(nbl::asset::E_FORMAT format, uint32_t layers = 0u);

		//
		bool Renderer::traceBounce(uint32_t & inoutRayCount);

		//
		const nbl::ext::MitsubaLoader::CMitsubaMetadata* m_globalMeta = nullptr;

		// "constants"
		bool m_useDenoiser;

		// managers
		nbl::video::IVideoDriver* m_driver;

		nbl::asset::IAssetManager* m_assetManager;
		nbl::scene::ISceneManager* m_smgr;

		nbl::core::smart_refctd_ptr<nbl::ext::RadeonRays::Manager> m_rrManager;
#ifdef _NBL_BUILD_OPTIX_
		nbl::core::smart_refctd_ptr<nbl::ext::OptiX::Manager> m_optixManager;
		CUstream m_cudaStream;
		nbl::core::smart_refctd_ptr<nbl::ext::OptiX::IContext> m_optixContext;
#endif


		// persistent (intialized in constructor
		nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> m_rayCountBuffer,m_littleDownloadBuffer;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> m_cullDSLayout;
		nbl::core::smart_refctd_ptr<const nbl::video::IGPUDescriptorSetLayout> m_perCameraRasterDSLayout;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> m_rasterInstanceDataDSLayout,m_additionalGlobalDSLayout,m_commonRaytracingDSLayout;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> m_raygenDSLayout,m_closestHitDSLayout,m_resolveDSLayout;
		nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline> m_visibilityBufferFillPipeline;

		nbl::core::smart_refctd_ptr<nbl::video::IGPUPipelineLayout> m_cullPipelineLayout;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUPipelineLayout> m_raygenPipelineLayout;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUPipelineLayout> m_closestHitPipelineLayout;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUPipelineLayout> m_resolvePipelineLayout;
		
		nbl::core::smart_refctd_ptr<IGPUSpecializedShader> m_cullGPUShader;
		nbl::core::smart_refctd_ptr<IGPUSpecializedShader> m_raygenGPUShader;
		nbl::core::smart_refctd_ptr<IGPUSpecializedShader> m_closestHitGPUShader;
		nbl::core::smart_refctd_ptr<IGPUSpecializedShader> m_resolveGPUShader;

		// scene specific data
		nbl::core::vector<::RadeonRays::Shape*> rrShapes;
		nbl::core::vector<::RadeonRays::Shape*> rrInstances;

		nbl::core::matrix3x4SIMD m_prevView;
		nbl::core::matrix4x3 m_prevCamTform;
		nbl::core::aabbox3df m_sceneBound;
		uint32_t m_framesDispatched;
		vec2 m_rcpPixelSize;
		uint64_t m_totalRaysCast;
		StaticViewData_t m_staticViewData;
		RaytraceShaderCommonData_t m_raytraceCommonData;

		nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> m_indexBuffer;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> m_indirectDrawBuffers[2];
		struct MDICall
		{
			uint32_t mdiOffset,mdiCount;
		};
		nbl::core::vector<MDICall> m_mdiDrawCalls;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_cullDS;
		CullShaderData_t m_cullPushConstants;
		uint32_t m_cullWorkGroups;

		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_perCameraRasterDS;
		
		nbl::core::smart_refctd_ptr<nbl::video::IGPUComputePipeline> m_cullPipeline,m_raygenPipeline,m_closestHitPipeline,m_resolvePipeline;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_globalBackendDataDS,m_additionalGlobalDS;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_commonRaytracingDS[2];
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_rasterInstanceDataDS,m_raygenDS,m_resolveDS;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_closestHitDS[2];
		uint32_t m_raygenWorkGroups[2];

		struct InteropBuffer
		{
			nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> buffer;
			std::pair<::RadeonRays::Buffer*, cl_mem> asRRBuffer = { nullptr,0u };
		};
		InteropBuffer m_rayBuffer[2];
		InteropBuffer m_intersectionBuffer[2];
		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> m_accumulation,m_tonemapOutput;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> m_albedoAcc,m_albedoRslv;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> m_normalAcc,m_normalRslv;
		nbl::video::IFrameBuffer* m_visibilityBuffer,* m_colorBuffer;
		
		// Resources used for blending environmental maps
		nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline> blendEnvPipeline;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> blendEnvDescriptorSet;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUMeshBuffer> blendEnvMeshBuffer;

		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> m_finalEnvmap;

		std::future<bool> compileShadersFuture;

	#ifdef _NBL_BUILD_OPTIX_
		nbl::core::smart_refctd_ptr<nbl::ext::OptiX::IDenoiser> m_denoiser;
		OptixDenoiserSizes m_denoiserMemReqs;
		nbl::cuda::CCUDAHandler::GraphicsAPIObjLink<nbl::video::IGPUBuffer> m_denoiserInputBuffer,m_denoiserStateBuffer,m_denoisedBuffer,m_denoiserScratchBuffer;

		enum E_DENOISER_INPUT
		{
			EDI_COLOR,
			EDI_ALBEDO,
			EDI_NORMAL,
			EDI_COUNT
		};
		OptixImage2D m_denoiserOutput;
		OptixImage2D m_denoiserInputs[EDI_COUNT];
	#endif
};

#endif
