#ifndef _EXTRA_CRAP_INCLUDED_
#define _EXTRA_CRAP_INCLUDED_

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
		#ifdef __cplusplus
			#undef mat4
			#undef mat4x3
		#endif

		// No 8k yet, too many rays to store
		_IRR_STATIC_INLINE_CONSTEXPR uint32_t MaxResolution[2] = {7680/2,4320/2};


		Renderer(irr::video::IVideoDriver* _driver, irr::asset::IAssetManager* _assetManager, irr::scene::ISceneManager* _smgr, irr::core::smart_refctd_ptr<irr::video::IGPUDescriptorSet>&& globalBackendDataDS, bool useDenoiser = true);

		void init(	const irr::asset::SAssetBundle& meshes,
					bool isCameraRightHanded,
					irr::core::smart_refctd_ptr<irr::asset::ICPUBuffer>&& sampleSequence,
					uint32_t rayBufferSize=(sizeof(::RadeonRays::ray)*2u+sizeof(uint32_t)*2u)*MaxResolution[0]*MaxResolution[1]); // 2 samples for MIS, TODO: compute default buffer size

		void deinit();

		void render(irr::ITimer* timer);

		bool isRightHanded() { return m_rightHanded; }

		auto* getColorBuffer() { return m_colorBuffer; }

		const auto& getSceneBound() const { return m_sceneBound; }

		uint64_t getTotalSamplesComputed() const { return static_cast<uint64_t>(m_samplesComputedPerPixel)*static_cast<uint64_t>(m_rayCountPerDispatch)/m_samplesPerPixelPerDispatch; }


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

#if TODO
		core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createDS2layoutCompost(bool useDenoiser, core::smart_refctd_ptr<video::IGPUSampler>& nearstSampler);
		core::smart_refctd_ptr<video::IGPUDescriptorSet> createDS2Compost(bool useDenoiser,
			core::smart_refctd_ptr<video::IGPUSampler>& nearestSampler
		);
		core::smart_refctd_ptr<video::IGPUPipelineLayout> createLayoutCompost();

		core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createDS2layoutRaygen(core::smart_refctd_ptr<video::IGPUSampler>& nearstSampler);
		core::smart_refctd_ptr<video::IGPUDescriptorSet> createDS2Raygen(core::smart_refctd_ptr<video::IGPUSampler>& nearstSampler);
		core::smart_refctd_ptr<video::IGPUPipelineLayout> createLayoutRaygen();
#endif

		const bool m_useDenoiser;

        irr::video::IVideoDriver* m_driver;

		irr::asset::IAssetManager* m_assetManager;
		irr::scene::ISceneManager* m_smgr;

		irr::core::smart_refctd_ptr<irr::ext::RadeonRays::Manager> m_rrManager;

		irr::core::smart_refctd_ptr<irr::asset::ICPUSpecializedShader> m_visibilityBufferFillShaders[2];
		irr::core::smart_refctd_ptr<irr::asset::ICPUPipelineLayout> m_visibilityBufferFillPipelineLayoutCPU;
		irr::core::smart_refctd_ptr<irr::video::IGPUPipelineLayout> m_visibilityBufferFillPipelineLayoutGPU;
		irr::core::smart_refctd_ptr<const irr::video::IGPUDescriptorSetLayout> m_perCameraRasterDSLayout;

		irr::core::smart_refctd_ptr<irr::video::IGPUDescriptorSetLayout> m_cullDSLayout;
		irr::core::smart_refctd_ptr<irr::video::IGPUPipelineLayout> m_cullPipelineLayout;
		irr::core::smart_refctd_ptr<irr::video::IGPUComputePipeline> m_cullPipeline;
		
		irr::core::smart_refctd_ptr<irr::video::IGPUComputePipeline> m_raygenPipeline,m_resolvePipeline;


		irr::core::vectorSIMDf baseEnvColor;
		irr::core::aabbox3df m_sceneBound;
		uint32_t m_renderSize[2u];
		bool m_rightHanded;

		irr::core::smart_refctd_ptr<irr::video::IGPUBuffer> m_indirectDrawBuffers[2];
		struct MDICall
		{
			irr::asset::SBufferBinding<irr::video::IGPUBuffer> vertexBindings[irr::video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
			irr::core::smart_refctd_ptr<irr::video::IGPUBuffer> indexBuffer;
			irr::core::smart_refctd_ptr<irr::video::IGPURenderpassIndependentPipeline> pipeline;
			uint32_t mdiOffset,mdiCount;
		};
		irr::core::vector<MDICall> m_mdiDrawCalls;
		CullShaderData_t m_cullPushConstants;

		uint32_t m_lightCount;
		irr::core::smart_refctd_ptr<irr::video::IGPUBuffer> m_lightCDFBuffer;
		irr::core::smart_refctd_ptr<irr::video::IGPUBuffer> m_lightBuffer;
		irr::core::smart_refctd_ptr<irr::video::IGPUBuffer> m_lightRadianceBuffer;

		irr::core::smart_refctd_ptr<irr::video::IGPUDescriptorSet> m_globalBackendDataDS,m_cullDS,m_perCameraRasterDS; // TODO: do we need to keep track of this?


		irr::ext::RadeonRays::Manager::MeshBufferRRShapeCache rrShapeCache;
#if TODO
		irr::core::smart_refctd_ptr<irr::video::IGPUDescriptorSet> m_compostDS2;
		irr::core::smart_refctd_ptr<irr::video::IGPUPipelineLayout> m_compostLayout;
		irr::core::smart_refctd_ptr<irr::video::IGPUComputePipeline> m_compostPipeline;

		uint32_t m_raygenWorkGroups[2];
		uint32_t m_resolveWorkGroups[2];

		irr::core::smart_refctd_ptr<irr::video::IGPUDescriptorSet> m_raygenDS2;
		irr::core::smart_refctd_ptr<irr::video::IGPUPipelineLayout> m_raygenLayout;

		irr::ext::RadeonRays::Manager::MeshNodeRRInstanceCache rrInstances;
#endif
		struct InteropBuffer
		{
			irr::core::smart_refctd_ptr<irr::video::IGPUBuffer> buffer;
			std::pair<::RadeonRays::Buffer*,cl_mem> asRRBuffer;
		};
		InteropBuffer m_rayCountBuffer,m_rayBuffer,m_intersectionBuffer;


		enum E_VISIBILITY_BUFFER_ATTACHMENT
		{
			EVBA_DEPTH,
			EVBA_OBJECTID_AND_TRIANGLEID_AND_FRONTFACING,
			// TODO: Once we get geometry packer V2 (virtual geometry) no need for these buffers actually (might want/need a barycentric buffer)
			EVBA_NORMALS,
			EVBA_UV_COORDINATES,
			EVBA_COUNT
		};
		irr::core::smart_refctd_ptr<irr::video::IGPUImageView> m_visibilityBufferAttachments[EVBA_COUNT];

		uint32_t m_maxSamples, m_samplesPerPixelPerDispatch, m_rayCountPerDispatch;
		uint32_t m_framesDone, m_samplesComputedPerPixel;
		irr::core::smart_refctd_ptr<irr::video::IGPUBufferView> m_sampleSequence;
		irr::core::smart_refctd_ptr<irr::video::IGPUImageView> m_scrambleTexture;

		irr::core::smart_refctd_ptr<irr::video::IGPUImageView> m_accumulation, m_tonemapOutput;
		irr::video::IFrameBuffer* m_visibilityBuffer,* m_colorBuffer,* tmpTonemapBuffer;

	#ifdef _IRR_BUILD_OPTIX_
		irr::core::smart_refctd_ptr<irr::ext::OptiX::Manager> m_optixManager;
		CUstream m_cudaStream;
		irr::core::smart_refctd_ptr<irr::ext::OptiX::IContext> m_optixContext;
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
