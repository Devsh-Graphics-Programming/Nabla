#ifndef _RENDERER_INCLUDED_
#define _RENDERER_INCLUDED_

#include "irrlicht.h"

#include "irr/ext/RadeonRays/RadeonRays.h"
// pesky leaking defines
#undef PI

#include "irr/ext/MitsubaLoader/CMitsubaLoader.h"

#include <ISceneManager.h>

#ifdef _IRR_BUILD_OPTIX_
#include "irr/ext/OptiX/Manager.h"
#endif


class Renderer : public irr::core::IReferenceCounted, public irr::core::InterfaceUnmovable
{
    public:
		#include "../drawCommon.glsl"
		#include "../raytraceCommon.glsl"
		#ifdef __cplusplus
			#undef uint
			#undef mat4
			#undef mat4x3
		#endif

		// No 8k yet, too many rays to store
		_IRR_STATIC_INLINE_CONSTEXPR uint32_t MaxResolution[2] = {7680/2,4320/2};


		Renderer(irr::video::IVideoDriver* _driver, irr::asset::IAssetManager* _assetManager, irr::scene::ISceneManager* _smgr, bool useDenoiser = true);

		void init(	const irr::asset::SAssetBundle& meshes, irr::core::smart_refctd_ptr<irr::asset::ICPUBuffer>&& sampleSequence,
					uint32_t rayBufferSize=(sizeof(::RadeonRays::ray)+sizeof(::RadeonRays::Intersection))*2u*MaxResolution[0]*MaxResolution[1]); // 2 samples for MIS, TODO: compute default buffer size

		void deinit();

		void render(irr::ITimer* timer);

		auto* getColorBuffer() { return m_colorBuffer; }

		const auto& getSceneBound() const { return m_sceneBound; }

		uint64_t getTotalSamplesComputed() const
		{
			const auto samplesPerDispatch = static_cast<uint64_t>(m_staticViewData.samplesPerRowPerDispatch*m_staticViewData.imageDimensions.y);
			const auto framesDispatched = static_cast<uint64_t>(m_raytraceCommonData.framesDispatched);
			return framesDispatched*samplesPerDispatch;
		}


		_IRR_STATIC_INLINE_CONSTEXPR uint32_t MaxDimensions = 6u;
    protected:
        ~Renderer();

		struct InitializationData
		{
			InitializationData() : lights(),lightRadiances(),lightCDF(),globalMeta(nullptr) {}
			InitializationData(InitializationData&& other) : InitializationData()
			{
				operator=(std::move(other));
			}
			~InitializationData() {lightCDF.~vector(); }

			inline InitializationData& operator=(InitializationData&& other)
			{
				lights = std::move(other.lights);
				lightRadiances = std::move(other.lightRadiances);
				lightCDF = std::move(other.lightCDF);
				globalMeta = other.globalMeta;
				return *this;
			}


			irr::core::vector<SLight> lights;
			irr::core::vector<irr::core::vectorSIMDf> lightRadiances;
			union
			{
				irr::core::vector<float> lightPDF;
				irr::core::vector<uint32_t> lightCDF;
			};
			const irr::ext::MitsubaLoader::CGlobalMitsubaMetadata* globalMeta = nullptr;
		};
		InitializationData initSceneObjects(const irr::asset::SAssetBundle& meshes);
		void initSceneNonAreaLights(InitializationData& initData);
		void finalizeScene(InitializationData& initData);

		irr::core::smart_refctd_ptr<irr::video::IGPUImageView> createScreenSizedTexture(irr::asset::E_FORMAT format);


		// constants
		const bool m_useDenoiser;

		// managers
        irr::video::IVideoDriver* m_driver;

		irr::asset::IAssetManager* m_assetManager;
		irr::scene::ISceneManager* m_smgr;

		irr::core::smart_refctd_ptr<irr::ext::RadeonRays::Manager> m_rrManager;
#ifdef _IRR_BUILD_OPTIX_
		irr::core::smart_refctd_ptr<irr::ext::OptiX::Manager> m_optixManager;
		CUstream m_cudaStream;
		irr::core::smart_refctd_ptr<irr::ext::OptiX::IContext> m_optixContext;
#endif


		// persistent (intialized in constructor
		irr::core::smart_refctd_ptr<irr::video::IGPUDescriptorSetLayout> m_cullDSLayout;
		irr::core::smart_refctd_ptr<irr::video::IGPUPipelineLayout> m_cullPipelineLayout;
		irr::core::smart_refctd_ptr<irr::video::IGPUComputePipeline> m_cullPipeline;

		irr::core::smart_refctd_ptr<irr::asset::ICPUSpecializedShader> m_visibilityBufferFillShaders[2];
		irr::core::smart_refctd_ptr<irr::asset::ICPUPipelineLayout> m_visibilityBufferFillPipelineLayoutCPU;
		irr::core::smart_refctd_ptr<irr::video::IGPUPipelineLayout> m_visibilityBufferFillPipelineLayoutGPU;
		irr::core::smart_refctd_ptr<irr::video::IGPUDescriptorSetLayout> m_perCameraRasterDSLayout;

		irr::core::smart_refctd_ptr<irr::video::IGPUDescriptorSetLayout> m_commonRaytracingDSLayout, m_raygenDSLayout, m_resolveDSLayout;


		// scene specific data
		irr::ext::RadeonRays::MockSceneManager m_mock_smgr;
		irr::ext::RadeonRays::Manager::MeshBufferRRShapeCache rrShapeCache;
		irr::ext::RadeonRays::Manager::NblInstanceRRInstanceCache rrInstances;

		irr::core::aabbox3df m_sceneBound;
		uint32_t m_maxRaysPerDispatch;
		StaticViewData_t m_staticViewData;
		RaytraceShaderCommonData_t m_raytraceCommonData;

		irr::core::smart_refctd_ptr<irr::video::IGPUBuffer> m_indirectDrawBuffers[2];
		struct MDICall
		{
			irr::asset::SBufferBinding<irr::video::IGPUBuffer> vertexBindings[irr::video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
			irr::core::smart_refctd_ptr<irr::video::IGPUBuffer> indexBuffer;
			irr::core::smart_refctd_ptr<irr::video::IGPURenderpassIndependentPipeline> pipeline;
			uint32_t mdiOffset, mdiCount;
		};
		irr::core::vector<MDICall> m_mdiDrawCalls;
		irr::core::smart_refctd_ptr<irr::video::IGPUDescriptorSet> m_cullDS;
		CullShaderData_t m_cullPushConstants;
		uint32_t m_cullWorkGroups;

		irr::core::smart_refctd_ptr<irr::video::IGPUDescriptorSet> m_perCameraRasterDS;
		
		irr::core::smart_refctd_ptr<irr::video::IGPUPipelineLayout> m_raygenPipelineLayout, m_resolvePipelineLayout;
		irr::core::smart_refctd_ptr<irr::video::IGPUComputePipeline> m_raygenPipeline, m_resolvePipeline;
		irr::core::smart_refctd_ptr<irr::video::IGPUDescriptorSet> m_globalBackendDataDS,m_commonRaytracingDS,m_raygenDS;
		uint32_t m_raygenWorkGroups[2];

		struct InteropBuffer
		{
			irr::core::smart_refctd_ptr<irr::video::IGPUBuffer> buffer;
			std::pair<::RadeonRays::Buffer*, cl_mem> asRRBuffer = { nullptr,0u };
		};
		InteropBuffer m_rayCountBuffer,m_rayBuffer,m_intersectionBuffer;

		irr::core::smart_refctd_ptr<irr::video::IGPUDescriptorSet> m_resolveDS;
		uint32_t m_resolveWorkGroups[2];

		irr::core::smart_refctd_ptr<irr::video::IGPUImageView> m_accumulation,m_tonemapOutput;
		irr::video::IFrameBuffer* m_visibilityBuffer,* m_colorBuffer,* tmpTonemapBuffer;

	#ifdef _IRR_BUILD_OPTIX_
		irr::core::smart_refctd_ptr<irr::ext::OptiX::IDenoiser> m_denoiser;
		OptixDenoiserSizes m_denoiserMemReqs;
		irr::cuda::CCUDAHandler::GraphicsAPIObjLink<irr::video::IGPUBuffer> m_denoiserInputBuffer,m_denoiserStateBuffer,m_denoisedBuffer,m_denoiserScratchBuffer;

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
