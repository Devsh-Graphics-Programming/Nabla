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
		struct alignas(16) SLight
		{
			SLight() {}
			SLight(const SLight& other)
			{
				std::copy(other.strengthFactor,other.strengthFactor+3u,strengthFactor);
				OBB.data = other.OBB.data;
			}
			~SLight() {}

			inline SLight& operator=(SLight&& other)
			{
				std::swap_ranges(strengthFactor,strengthFactor+3u,other.strengthFactor);
				std::swap(OBB.data, other.OBB.data);

				return *this;
			}

			void setFactor(const irr::core::vectorSIMDf& _strengthFactor)
			{
				for (auto i=0u; i<3u; i++)
					strengthFactor[i] = _strengthFactor[i];
			}
#if TODO
			//! This is according to Rec.709 colorspace
			inline float getFactorLuminosity() const
			{
				float rec709LumaCoeffs[] = {0.2126f, 0.7152, 0.0722};

				//! TODO: More color spaces!
				float* colorSpaceLumaCoeffs = rec709LumaCoeffs;

				float luma = strengthFactor[0] * colorSpaceLumaCoeffs[0];
				luma += strengthFactor[1] * colorSpaceLumaCoeffs[1];
				luma += strengthFactor[2] * colorSpaceLumaCoeffs[2];
				return luma;
			}

			inline float computeAreaUnderTransform(irr::core::vectorSIMDf differentialElementCrossProduct) const
			{
				analytical.transformCofactors.mulSub3x3WithNx1(differentialElementCrossProduct);
				return irr::core::length(differentialElementCrossProduct).x;
			}

			inline float computeFlux(float triangulizationArea) const // also known as lumens
			{
				const auto unitHemisphereArea = 2.f*irr::core::PI<float>();
				const auto unitSphereArea = 2.f*unitHemisphereArea;

				float lightFlux = unitHemisphereArea*getFactorLuminosity();
				switch (type)
				{
					case ET_ELLIPSOID:
						_IRR_FALLTHROUGH;
					case ET_TRIANGLE:
						lightFlux *= triangulizationArea;
						break;
					default:
						assert(false);
						break;
				}
				return lightFlux;
			}

			static inline SLight createFromTriangle(const irr::core::vectorSIMDf& _strengthFactor, const CachedTransform& precompTform, const irr::core::vectorSIMDf* v, float* outArea=nullptr)
			{
				SLight triLight;
				triLight.type = ET_TRIANGLE;
				triLight.setFactor(_strengthFactor);
				triLight.analytical = precompTform;

				float triangleArea = 0.5f*triLight.computeAreaUnderTransform(irr::core::cross(v[1]-v[0],v[2]-v[0]));
				if (outArea)
					*outArea = triangleArea;

				for (auto k=0u; k<3u; k++)
					precompTform.transform.transformVect(triLight.triangle.vertices[k], v[k]);
				triLight.triangle.vertices[1] -= triLight.triangle.vertices[0];
				triLight.triangle.vertices[2] -= triLight.triangle.vertices[0];
				// always flip the handedness so normal points inwards (need negative normal for differential area optimization)
				if (precompTform.transform.getPseudoDeterminant().x>0.f)
					std::swap(triLight.triangle.vertices[2], triLight.triangle.vertices[1]);

				// don't do any flux magic yet

				return triLight;
			}
#endif
			//! type is second member due to alignment issues
			union OBB_t
			{
				OBB_t() : data() {}
				~OBB_t() {}
				OBB_t(const OBB_t& rhs) : data(rhs.data) {}

				irr::core::matrix3x4SIMD data;
				struct
				{
					float e1_x, e2_x, e3_x, offset_x;
					float e1_y, e2_y, e3_y, offset_y;
					float e1_z, e2_z, e3_z, offset_z;
				};
			} OBB;
			//! different lights use different measures of their strength (this already has the reciprocal of the light PDF factored in)
			alignas(16) float strengthFactor[3];
		};
		static_assert(sizeof(SLight)==64u,"Can't keep alignment straight!");

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


		irr::core::aabbox3df m_sceneBound;
		uint32_t m_renderSize[2u];
		bool m_rightHanded;

		irr::core::smart_refctd_ptr<irr::video::IGPUDescriptorSet> m_globalBackendDataDS;

		irr::ext::RadeonRays::Manager::MeshBufferRRShapeCache rrShapeCache;
#if TODO
		irr::core::smart_refctd_ptr<irr::video::IGPUDescriptorSet> m_compostDS2;
		irr::core::smart_refctd_ptr<irr::video::IGPUPipelineLayout> m_compostLayout;
		irr::core::smart_refctd_ptr<irr::video::IGPUComputePipeline> m_compostPipeline;

		uint32_t m_raygenWorkGroups[2];
		uint32_t m_resolveWorkGroups[2];
		irr::core::smart_refctd_ptr<irr::video::IGPUBuffer> m_rayBuffer;
		irr::core::smart_refctd_ptr<irr::video::IGPUBuffer> m_intersectionBuffer;
		irr::core::smart_refctd_ptr<irr::video::IGPUBuffer> m_rayCountBuffer;
		std::pair<::RadeonRays::Buffer*,cl_mem> m_rayBufferAsRR;
		std::pair<::RadeonRays::Buffer*,cl_mem> m_intersectionBufferAsRR;
		std::pair<::RadeonRays::Buffer*,cl_mem> m_rayCountBufferAsRR;

		irr::core::smart_refctd_ptr<irr::video::IGPUDescriptorSet> m_raygenDS2;
		irr::core::smart_refctd_ptr<irr::video::IGPUPipelineLayout> m_raygenLayout;
		irr::core::smart_refctd_ptr<irr::video::IGPUComputePipeline> m_raygenPipeline;

		irr::core::vector<irr::core::smart_refctd_ptr<irr::scene::IMeshSceneNode> > nodes;
		irr::ext::RadeonRays::Manager::MeshNodeRRInstanceCache rrInstances;

		irr::core::vectorSIMDf constantClearColor;
#endif
		uint32_t m_lightCount;
		irr::core::smart_refctd_ptr<irr::video::IGPUBuffer> m_lightCDFBuffer;
		irr::core::smart_refctd_ptr<irr::video::IGPUBuffer> m_lightBuffer;
		irr::core::smart_refctd_ptr<irr::video::IGPUBuffer> m_lightRadianceBuffer;

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
