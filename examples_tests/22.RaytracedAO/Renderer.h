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

		// No 8k yet, too many rays to store
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t MaxResolution[2] = {7680/2,4320/2};


		Renderer(nbl::video::IVideoDriver* _driver, nbl::asset::IAssetManager* _assetManager, nbl::scene::ISceneManager* _smgr, bool useDenoiser = true);

		void init(	const nbl::asset::SAssetBundle& meshes, nbl::core::smart_refctd_ptr<nbl::asset::ICPUBuffer>&& sampleSequence,
					uint32_t rayBufferSize=(sizeof(::RadeonRays::ray)+sizeof(::RadeonRays::Intersection))*2u*MaxResolution[0]*MaxResolution[1]); // 2 samples for MIS, TODO: compute default buffer size

		void deinit();

		void render(nbl::ITimer* timer);

		auto* getColorBuffer() { return m_colorBuffer; }

		const auto& getSceneBound() const { return m_sceneBound; }

		uint64_t getTotalSamplesComputed() const
		{
			const auto samplesPerDispatch = static_cast<uint64_t>(m_staticViewData.samplesPerRowPerDispatch*m_staticViewData.imageDimensions.y);
			const auto framesDispatched = static_cast<uint64_t>(m_raytraceCommonData.framesDispatched);
			return framesDispatched*samplesPerDispatch;
		}


		_NBL_STATIC_INLINE_CONSTEXPR uint32_t MaxDimensions = 6u;
		static const float AntiAliasingSequence[4096][2];
    protected:
        ~Renderer();

		struct InitializationData
		{
			InitializationData() : mdiFirstIndices(), lights(),lightRadiances(),lightCDF(),globalMeta(nullptr) {}
			InitializationData(InitializationData&& other) : InitializationData()
			{
				operator=(std::move(other));
			}
			~InitializationData() {lightCDF.~vector(); }

			inline InitializationData& operator=(InitializationData&& other)
			{
				mdiFirstIndices = std::move(other.mdiFirstIndices);
				lights = std::move(other.lights);
				lightRadiances = std::move(other.lightRadiances);
				lightCDF = std::move(other.lightCDF);
				globalMeta = other.globalMeta;
				return *this;
			}

			nbl::core::vector<uint32_t> mdiFirstIndices;
			nbl::core::vector<SLight> lights;
			nbl::core::vector<nbl::core::vectorSIMDf> lightRadiances;
			union
			{
				nbl::core::vector<float> lightPDF;
				nbl::core::vector<uint32_t> lightCDF;
			};
			const nbl::ext::MitsubaLoader::CMitsubaMetadata* globalMeta = nullptr;
		};
		InitializationData initSceneObjects(const nbl::asset::SAssetBundle& meshes);
		void initSceneNonAreaLights(InitializationData& initData);
		void finalizeScene(InitializationData& initData);

		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> createScreenSizedTexture(nbl::asset::E_FORMAT format);


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
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> m_cullDSLayout;
		nbl::core::smart_refctd_ptr<const nbl::video::IGPUDescriptorSetLayout> m_perCameraRasterDSLayout;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> m_rasterInstanceDataDSLayout,m_additionalGlobalDSLayout,m_commonRaytracingDSLayout,m_raygenDSLayout,m_resolveDSLayout;
		nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline> m_visibilityBufferFillPipeline;


		// scene specific data
		nbl::core::vector<::RadeonRays::Shape*> rrShapes;
		nbl::core::vector<::RadeonRays::Shape*> rrInstances;

		nbl::core::matrix3x4SIMD m_prevView;
		nbl::core::aabbox3df m_sceneBound;
		uint32_t m_maxRaysPerDispatch;
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
		
		nbl::core::smart_refctd_ptr<nbl::video::IGPUPipelineLayout> m_cullPipelineLayout, m_raygenPipelineLayout, m_resolvePipelineLayout;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUComputePipeline> m_cullPipeline, m_raygenPipeline, m_resolvePipeline;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_globalBackendDataDS,m_rasterInstanceDataDS,m_additionalGlobalDS,m_commonRaytracingDS,m_raygenDS;
		uint32_t m_raygenWorkGroups[2];

		struct InteropBuffer
		{
			nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> buffer;
			std::pair<::RadeonRays::Buffer*, cl_mem> asRRBuffer = { nullptr,0u };
		};
		InteropBuffer m_rayCountBuffer,m_rayBuffer,m_intersectionBuffer;

		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_resolveDS;
		uint32_t m_resolveWorkGroups[2];

		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> m_accumulation,m_tonemapOutput;
		nbl::video::IFrameBuffer* m_visibilityBuffer,* m_colorBuffer,* tmpTonemapBuffer;

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
