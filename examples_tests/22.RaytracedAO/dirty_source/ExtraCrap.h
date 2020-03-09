#ifndef _EXTRA_CRAP_INCLUDED_
#define _EXTRA_CRAP_INCLUDED_

#include "irrlicht.h"

#include "../../ext/RadeonRays/RadeonRays.h"
// pesky leaking defines
#undef PI

#ifdef _IRR_BUILD_OPTIX_
#include "../../ext/OptiX/Manager.h"
#endif


class Renderer : public irr::core::IReferenceCounted, public irr::core::InterfaceUnmovable
{
    public:
		struct alignas(16) SLight
		{
			enum E_TYPE : uint32_t
			{
				ET_ELLIPSOID,
				ET_TRIANGLE,
				ET_COUNT
			};
			struct CachedTransform
			{
				CachedTransform(const irr::core::matrix3x4SIMD& tform) : transform(tform)
				{
					auto tmp4 = irr::core::matrix4SIMD(transform.getSub3x3TransposeCofactors());
					transformCofactors = irr::core::transpose(tmp4).extractSub3x4();
				}

				irr::core::matrix3x4SIMD transform;
				irr::core::matrix3x4SIMD transformCofactors;
			};


			SLight() : type(ET_COUNT) {}
			SLight(const SLight& other) : type(other.type)
			{
				std::copy(other.strengthFactor,other.strengthFactor+3u,strengthFactor);
				if (type == ET_TRIANGLE)
					std::copy(other.triangle.vertices, other.triangle.vertices+3u, triangle.vertices);
				else
				{
					analytical.transform = other.analytical.transform;
					analytical.transformCofactors = other.analytical.transformCofactors;
				}
			}
			~SLight() {}

			inline SLight& operator=(SLight&& other)
			{
				std::swap_ranges(strengthFactor,strengthFactor+3u,other.strengthFactor);
				auto a = other.analytical;
				auto t = other.triangle;
				if (type!=ET_TRIANGLE)
					other.analytical = analytical;
				else
					other.triangle = triangle;
				if (other.type)
					analytical = a;
				else
					triangle = t;
				std::swap(type, other.type);

				return *this;
			}

			void setFactor(const irr::core::vectorSIMDf& _strengthFactor)
			{
				for (auto i=0u; i<3u; i++)
					strengthFactor[i] = _strengthFactor[i];
			}

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

			//! different lights use different measures of their strength (this already has the reciprocal of the light PDF factored in)
			alignas(16) float strengthFactor[3];
			//! type is second member due to alignment issues
			E_TYPE type;
			//! useful for analytical shapes
			union
			{
				CachedTransform analytical;
				struct Triangle
				{
					irr::core::vectorSIMDf padding[3];
					irr::core::vectorSIMDf vertices[3];
				} triangle;
			};
			/*
			union
			{
				AreaSphere sphere;
			};
			*/
		};
		static_assert(sizeof(SLight)==112u,"Can't keep alignment straight!");

		// No 8k yet, too many rays to store
		_IRR_STATIC_INLINE_CONSTEXPR uint32_t MaxResolution[2] = {7680/2,4320/2};


		Renderer(irr::video::IVideoDriver* _driver, irr::asset::IAssetManager* _assetManager, irr::scene::ISceneManager* _smgr, bool useDenoiser = true);

		void init(	const irr::asset::SAssetBundle& meshes,
					bool isCameraRightHanded,
					irr::core::smart_refctd_ptr<irr::asset::ICPUBuffer>&& sampleSequence,
					uint32_t rayBufferSize=(sizeof(::RadeonRays::ray)*2u+sizeof(uint32_t)*2u)*MaxResolution[0]*MaxResolution[1]); // 2 samples for MIS

		void deinit();

		void render();

		bool isRightHanded() { return m_rightHanded; }

		auto* getColorBuffer() { return m_colorBuffer; }

		const auto& getSceneBound() const { return sceneBound; }

		uint64_t getTotalSamplesComputed() const { return static_cast<uint64_t>(m_samplesComputed)*static_cast<uint64_t>(m_rayCount)/m_samplesPerDispatch; }


		_IRR_STATIC_INLINE_CONSTEXPR uint32_t MaxDimensions = 4u;
    protected:
        ~Renderer();


        irr::video::IVideoDriver* m_driver;

		irr::asset::IAssetManager* m_assetManager;
		irr::scene::ISceneManager* m_smgr;

		irr::core::vector<std::array<irr::core::vector3df_SIMD,3> > m_precomputedGeodesic;

		irr::core::smart_refctd_ptr<irr::ext::RadeonRays::Manager> m_rrManager;

		bool m_rightHanded;

		irr::core::smart_refctd_ptr<irr::video::ITextureBufferObject> m_sampleSequence;
		irr::core::smart_refctd_ptr<irr::video::ITexture> m_scrambleTexture;

		irr::video::E_MATERIAL_TYPE nonInstanced;
		uint32_t m_raygenProgram, m_compostProgram;
		irr::core::smart_refctd_ptr<irr::video::ITexture> m_depth,m_albedo,m_normals,m_lightIndex,m_accumulation,m_tonemapOutput;
		irr::video::IFrameBuffer* m_colorBuffer,* m_gbuffer,* tmpTonemapBuffer;

		uint32_t m_maxSamples;
		uint32_t m_raygenWorkGroups[2];
		uint32_t m_resolveWorkGroups[2];
		uint32_t m_samplesPerDispatch;
		uint32_t m_samplesComputed;
		uint32_t m_rayCount;
		uint32_t m_framesDone;
		irr::core::smart_refctd_ptr<irr::video::IGPUBuffer> m_rayBuffer;
		irr::core::smart_refctd_ptr<irr::video::IGPUBuffer> m_intersectionBuffer;
		irr::core::smart_refctd_ptr<irr::video::IGPUBuffer> m_rayCountBuffer;
		std::pair<::RadeonRays::Buffer*,cl_mem> m_rayBufferAsRR;
		std::pair<::RadeonRays::Buffer*,cl_mem> m_intersectionBufferAsRR;
		std::pair<::RadeonRays::Buffer*,cl_mem> m_rayCountBufferAsRR;

		irr::core::vector<irr::core::smart_refctd_ptr<irr::scene::IMeshSceneNode> > nodes;
		irr::core::aabbox3df sceneBound;
		irr::ext::RadeonRays::Manager::MeshBufferRRShapeCache rrShapeCache;
		irr::ext::RadeonRays::Manager::MeshNodeRRInstanceCache rrInstances;

		irr::core::vectorSIMDf constantClearColor;
		uint32_t m_lightCount;
		irr::core::smart_refctd_ptr<irr::video::IGPUBuffer> m_lightCDFBuffer;
		irr::core::smart_refctd_ptr<irr::video::IGPUBuffer> m_lightBuffer;
		irr::core::smart_refctd_ptr<irr::video::IGPUBuffer> m_lightRadianceBuffer;

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
			//EDI_NORMAL,
			//EDI_ALBEDO,
			EDI_COUNT
		};
		OptixImage2D m_denoiserOutput;
		OptixImage2D m_denoiserInputs[EDI_COUNT];
	#endif
};

#endif
