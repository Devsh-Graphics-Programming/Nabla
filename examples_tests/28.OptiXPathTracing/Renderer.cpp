#include <numeric>

#if 0
#include "Renderer.h"

#include "../../ext/ScreenShot/ScreenShot.h"

#include "../../ext/MitsubaLoader/CMitsubaLoader.h"

#define __C_CUDA_HANDLER_H__ // don't want CUDA declarations and defines to pollute here
#include "../source/Irrlicht/COpenGLBuffer.h"
#include "../source/Irrlicht/COpenGLTexture.h"
#include "../source/Irrlicht/COpenGLDriver.h"

using namespace irr;
using namespace irr::asset;
using namespace irr::video;
using namespace irr::scene;


const std::string raygenShaderExtensions = R"======(
#version 430 core
)======";

const std::string lightStruct = R"======(
#define SLight_ET_ELLIPSOID	0u
#define SLight_ET_TRIANGLE	1u
#define SLight_ET_COUNT		2u
struct SLight
{
	vec3 factor;
	uint data0;
	mat4x3 transform; // needs row_major qualifier
	mat3 transformCofactors;
};
uint SLight_extractType(in SLight light)
{
	return bitfieldExtract(light.data0,0,findMSB(SLight_ET_COUNT)+1);
}
)======";

const std::string raygenShader = R"======(
#define WORK_GROUP_DIM 16u
layout(local_size_x = WORK_GROUP_DIM, local_size_y = WORK_GROUP_DIM) in;
#define WORK_GROUP_SIZE (WORK_GROUP_DIM*WORK_GROUP_DIM)

// uniforms
layout(location = 0) uniform vec3 uCameraPos;
layout(location = 1) uniform float uDepthLinearizationConstant;
layout(location = 2) uniform mat4 uFrustumCorners;
layout(location = 3) uniform uvec2 uImageSize;
layout(location = 4) uniform uvec2 uSamples_ImageWidthSamples;
layout(location = 5) uniform uint uSamplesComputed;
layout(location = 6) uniform vec4 uImageSize2Rcp;

// image views
layout(binding = 0) uniform sampler2D depthbuf;
layout(binding = 1) uniform sampler2D normalbuf;
layout(binding = 2) uniform sampler2D albedobuf;
layout(binding = 3) uniform usamplerBuffer sampleSequence;
layout(binding = 4) uniform usampler2D scramblebuf;

// SSBOs
layout(binding = 0, std430) restrict writeonly buffer Rays
{
	RadeonRays_ray rays[];
};

layout(binding = 1, std430) restrict readonly buffer CumulativeLightPDF
{
	uint lightCDF[];
};

layout(binding = 2, std430, row_major) restrict readonly buffer Lights
{
	SLight light[];
};


#define kPI 3.14159265358979323846

vec3 decode(in vec2 enc)
{
	float ang = enc.x*kPI;
    return vec3(vec2(cos(ang),sin(ang))*sqrt(1.0-enc.y*enc.y), enc.y);
}


#define FLT_MIN -1.17549449e-38
#define FLT_MAX 3.402823466e+38

float linearizeZBufferVal(in float nonLinearZBufferVal)
{
	// 1-(Ax+B)/(Cx) = y
	// (Ax+B)/(Cx) = 1-y
	// x = B/(C(1-y)-A)
	// x = B/(C-A-Cy)
	// get back original Z: `row[2][3]/(row[3][2]-row[2][2]-y*row[3][2]) = x`
	// max Z: `B/(C-A)`
	// positive [0,1] Z: `B/(C-A-Cy)/(B/(C-A))`
	// positive [0,1] Z: `(C-A)/(C-A-Cy)`
	// positive [0,1] Z: `D/(D-Cy)`
    return 1.0/(uDepthLinearizationConstant*nonLinearZBufferVal+1.0);
}

float maxAbs1(in float val)
{
	return abs(val);
}
float maxAbs2(in vec2 val)
{
	vec2 v = abs(val);
	return max(v.x,v.y);
}
float maxAbs3(in vec3 val)
{
	vec3 v = abs(val);
	return max(max(v.x,v.y),v.z);
}

float ULP1(in float val, in uint accuracy)
{
	return uintBitsToFloat(floatBitsToUint(abs(val)) + accuracy)-val;
}
float ULP2(in vec2 val, in uint accuracy)
{
	float x = maxAbs2(val);
	return uintBitsToFloat(floatBitsToUint(x) + accuracy)-x;
}
float ULP3(in vec3 val, in uint accuracy)
{
	float x = maxAbs3(val);
	return uintBitsToFloat(floatBitsToUint(x) + accuracy)-x;
}



uint ugen_uniform_sample1(in uint dimension, in uint sampleIx, in uint scramble);
uvec2 ugen_uniform_sample2(in uint dimension, in uint sampleIx, in uint scramble);

vec2 gen_uniform_sample2(in uint dimension, in uint sampleIx, in uint scramble);


uint ugen_uniform_sample1(in uint dimension, in uint sampleIx, in uint scramble)
{
	return ugen_uniform_sample2(dimension,sampleIx,scramble).x;
}
uvec2 ugen_uniform_sample2(in uint dimension, in uint sampleIx, in uint scramble)
{
	uint address = (dimension>>1u)*MAX_SAMPLES+(sampleIx&(MAX_SAMPLES-1u));
	return texelFetch(sampleSequence,int(address)).xy^uvec2(scramble);
}

vec2 gen_uniform_sample2(in uint dimension, in uint sampleIx, in uint scramble)
{
	return vec2(ugen_uniform_sample2(dimension,sampleIx,scramble))/vec2(~0u);
}



// https://orbit.dtu.dk/files/126824972/onb_frisvad_jgt2012_v2.pdf
mat2x3 frisvad(in vec3 n)
{
	const float a = 1.0/(1.0 + n.z);
	const float b = -n.x*n.y*a;
	return (n.z<-0.9999999) ? mat2x3(vec3(0.0,-1.0,0.0),vec3(-1.0,0.0,0.0)):mat2x3(vec3(1.0-n.x*n.x*a, b, -n.x),vec3(b, 1.0-n.y*n.y*a, -n.y));
}


uint lower_bound(in uint key, uint size)
{
    uint low = 0u;

#define ITERATION \
	{\
        uint _half = size >> 1u; \
        uint other_half = size - _half; \
        uint probe = low + _half; \
        uint other_low = low + other_half; \
        uint v = lightCDF[probe]; \
        size = _half; \
        low = v<key ? other_low:low; \
	}

    while (size >= 8u)
	{
		ITERATION
		ITERATION
		ITERATION
    }

    while (size > 0u)
		ITERATION

#undef ITERATION

	return low;
}

uint upper_bound(in uint key, uint size)
{
    uint low = 0u;

#define ITERATION \
	{\
        uint _half = size >> 1u; \
        uint other_half = size - _half; \
        uint probe = low + _half; \
        uint other_low = low + other_half; \
        uint v = lightCDF[probe]; \
        size = _half; \
        low = v<=key ? other_low:low; \
	}

    while (size >= 8u)
	{
		ITERATION
		ITERATION
		ITERATION
    }

    while (size > 0u)
		ITERATION

#undef ITERATION

	return low;
}


vec3 light_sample(out vec3 incoming, in uint sampleIx, in uint scramble, inout float maxT, inout bool alive, in vec3 position, in vec3 normal)
{
	uint lightIDSample = ugen_uniform_sample1(0u,sampleIx,scramble);
	vec2 lightSurfaceSample = gen_uniform_sample2(2u,sampleIx,scramble);

	uint lightID = upper_bound(lightIDSample,uint(lightCDF.length()-1));

	SLight light = light[lightID];

#define SHADOW_RAY_LEN 0.93
	float factor; // 1.0/light_probability already baked into the light factor
	switch (SLight_extractType(light))
	{
		case SLight_ET_ELLIPSOID:
			lightSurfaceSample.x = lightSurfaceSample.x*2.0-1.0;
			{
				mat4x3 tform = light.transform;
				float equator = lightSurfaceSample.y*2.0*kPI;
				vec3 pointOnSurface = vec3(vec2(cos(equator),sin(equator))*sqrt(1.0-lightSurfaceSample.x*lightSurfaceSample.x),lightSurfaceSample.x);
	
				incoming = mat3(tform)*pointOnSurface+(tform[3]-position);
				float incomingInvLen = inversesqrt(dot(incoming,incoming));
				incoming *= incomingInvLen;

				maxT = SHADOW_RAY_LEN/incomingInvLen;

				factor = 4.0*kPI; // compensate for the domain of integration
				// don't normalize, length of the normal times determinant is very handy for differential area after a 3x3 matrix transform
				vec3 negLightNormal = light.transformCofactors*pointOnSurface;

				factor *= max(dot(negLightNormal,incoming),0.0)*incomingInvLen*incomingInvLen;
			}
			break;
		default: // SLight_ET_TRIANGLE:
			{
				vec3 pointOnSurface = transpose(light.transformCofactors)[0];
				vec3 shortEdge = transpose(light.transformCofactors)[1];
				vec3 longEdge = transpose(light.transformCofactors)[2];

				lightSurfaceSample.x = sqrt(lightSurfaceSample.x);

				pointOnSurface += (shortEdge*(1.0-lightSurfaceSample.y)+longEdge*lightSurfaceSample.y)*lightSurfaceSample.x;

				vec3 negLightNormal = cross(shortEdge,longEdge);

				incoming = pointOnSurface-position;
				float incomingInvLen = inversesqrt(dot(incoming,incoming));
				incoming *= incomingInvLen;

				maxT = SHADOW_RAY_LEN/incomingInvLen;

				factor = 0.5*max(dot(negLightNormal,incoming),0.0)*incomingInvLen*incomingInvLen;
			}
			break;
	}

	if (factor<0.0) // TODO: FLT_MIN
		alive = false;

	return light.factor*factor;
}

void main()
{
	uvec2 outputLocation = gl_GlobalInvocationID.xy;
	bool alive = all(lessThan(outputLocation,uImageSize));
	if (alive)
	{
		// TODO: accelerate texture fetching
		ivec2 uv = ivec2(outputLocation);
		float revdepth = texelFetch(depthbuf,uv,0).r;

		uint outputID = uSamples_ImageWidthSamples.x*outputLocation.x+uSamples_ImageWidthSamples.y*outputLocation.y;

		// unproject
		vec3 viewDir;
		vec3 position;
		{
			vec2 NDC = vec2(outputLocation)*uImageSize2Rcp.xy+uImageSize2Rcp.zw;
			viewDir = mix(uFrustumCorners[0]*NDC.x+uFrustumCorners[1],uFrustumCorners[2]*NDC.x+uFrustumCorners[3],NDC.yyyy).xyz;
			position = viewDir*linearizeZBufferVal(revdepth)+uCameraPos;
		}

		alive = revdepth>0.0;
		vec2 encNormal;
		if (alive)
			encNormal = texelFetch(normalbuf,uv,0).rg;

		uint scramble = texelFetch(scramblebuf,uv,0).r;

		RadeonRays_ray newray;
		newray.time = 0.0;
		newray.mask = alive ? -1:0;
		vec3 normal;
		if (alive)
			normal = decode(encNormal);
		for (uint i=0u; i<uSamples_ImageWidthSamples.x; i++)
		{
			vec4 bsdf = vec4(0.0,0.0,0.0,-1.0);
			float error = ULP1(uDepthLinearizationConstant,8u);

			newray.maxT = FLT_MAX;
			if (alive)
				bsdf.rgb = light_sample(newray.direction,uSamplesComputed+i,scramble,newray.maxT,alive,position,normal);
			if (alive)
				bsdf.rgb *= texelFetch(albedobuf,uv,0).rgb*max(dot(normal,newray.direction),0.0)/kPI;

			newray.origin = position+newray.direction*error/maxAbs3(newray.direction);
			newray._active = alive ? 1:0;
			newray.backfaceCulling = int(packHalf2x16(bsdf.ab));
			newray.useless_padding = int(packHalf2x16(bsdf.gr));

			// TODO: repack rays for coalescing
			rays[outputID+i] = newray;
		}
	}
}

)======";

const std::string compostShaderExtensions = R"======(
#version 430 core
)======";

const std::string compostShader = R"======(
#define WORK_GROUP_DIM 16u
layout(local_size_x = WORK_GROUP_DIM, local_size_y = WORK_GROUP_DIM) in;
#define WORK_GROUP_SIZE (WORK_GROUP_DIM*WORK_GROUP_DIM)

// uniforms
layout(location = 0) uniform uvec2 uImageSize;
layout(location = 1) uniform uvec2 uSamples_ImageWidthSamples;
layout(location = 2) uniform float uRcpFramesDone;

// image views
layout(binding = 0) uniform usampler2D lightIndex;
layout(binding = 0, rgba32f) restrict uniform image2D framebuffer;

// SSBOs
layout(binding = 0, std430) restrict readonly buffer Rays
{
	RadeonRays_ray rays[];
};
layout(binding = 1, std430) restrict buffer Queries
{
	int hit[];
};

layout(binding = 2, std430, row_major) restrict readonly buffer LightRadiances
{
	vec3 lightRadiance[]; // Watts / steriadian / steradian
};

void main()
{
	uvec2 outputLocation = gl_GlobalInvocationID.xy;
	bool alive = all(lessThan(outputLocation,uImageSize));
	if (alive)
	{
		ivec2 uv = ivec2(outputLocation);
		vec4 acc = vec4(0.0);
		if (uRcpFramesDone<1.0)
			acc = imageLoad(framebuffer,uv);

		uint baseID = uSamples_ImageWidthSamples.x*outputLocation.x+uSamples_ImageWidthSamples.y*outputLocation.y;
		vec3 color = vec3(0.0);
		for (uint i=0u; i<uSamples_ImageWidthSamples.x; i++)
		{
			uint rayID = baseID+i;
			if (hit[rayID]<0)
				color += vec4(unpackHalf2x16(rays[rayID].useless_padding),unpackHalf2x16(rays[rayID].backfaceCulling)).gra;//b;
			// hit buffer needs clearing
			hit[rayID] = -1;
		}
		// TODO: move the `div` to tonemapping shader (or ray gen actually, for fractional sampling)
		color *= 1.0/float(uSamples_ImageWidthSamples.x);

		// TODO: move this to tonemapping
		uint lightID = texelFetch(lightIndex,uv,0)[0];
		if (lightID!=0xdeadbeefu)
			color += lightRadiance[lightID];

		// TODO: optimize the color storage (RGB9E5/RGB19E7 anyone?)
		acc.rgb += (color-acc.rgb)*uRcpFramesDone;
		imageStore(framebuffer,uv,acc);
	}
}

)======";

inline GLuint createComputeShader(const std::string& source)
{
    GLuint program = COpenGLExtensionHandler::extGlCreateProgram();
	GLuint cs = COpenGLExtensionHandler::extGlCreateShader(GL_COMPUTE_SHADER);

	const char* tmp = source.c_str();
	COpenGLExtensionHandler::extGlShaderSource(cs, 1, const_cast<const char**>(&tmp), NULL);
	COpenGLExtensionHandler::extGlCompileShader(cs);

	// check for compilation errors
    GLint success;
    GLchar infoLog[0x200];
    COpenGLExtensionHandler::extGlGetShaderiv(cs, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        COpenGLExtensionHandler::extGlGetShaderInfoLog(cs, sizeof(infoLog), nullptr, infoLog);
        os::Printer::log("CS COMPILATION ERROR:\n", infoLog,ELL_ERROR);
        COpenGLExtensionHandler::extGlDeleteShader(cs);
        COpenGLExtensionHandler::extGlDeleteProgram(program);
        return 0;
	}

	COpenGLExtensionHandler::extGlAttachShader(program, cs);
	COpenGLExtensionHandler::extGlLinkProgram(program);

	//check linking errors
	success = 0;
    COpenGLExtensionHandler::extGlGetProgramiv(program, GL_LINK_STATUS, &success);
    if (success == GL_FALSE)
    {
        COpenGLExtensionHandler::extGlGetProgramInfoLog(program, sizeof(infoLog), nullptr, infoLog);
        os::Printer::log("CS LINK ERROR:\n", infoLog,ELL_ERROR);
        COpenGLExtensionHandler::extGlDeleteShader(cs);
        COpenGLExtensionHandler::extGlDeleteProgram(program);
        return 0;
    }

	return program;
}



class SimpleCallBack : public video::IShaderConstantSetCallBack
{
		Renderer* renderer;
		video::E_MATERIAL_TYPE currentMat;

		int32_t mvpUniformLocation[video::EMT_COUNT+2];
		int32_t nUniformLocation[video::EMT_COUNT+2];
		int32_t flipFacesUniformLocation[video::EMT_COUNT+2];
		int32_t colorUniformLocation[video::EMT_COUNT+2];
		int32_t nastyUniformLocation[video::EMT_COUNT+2];
		int32_t lightIDUniformLocation[video::EMT_COUNT+2];
		video::E_SHADER_CONSTANT_TYPE mvpUniformType[video::EMT_COUNT+2];
		video::E_SHADER_CONSTANT_TYPE nUniformType[video::EMT_COUNT+2];
		video::E_SHADER_CONSTANT_TYPE flipFacesUniformType[video::EMT_COUNT+2];
		video::E_SHADER_CONSTANT_TYPE colorUniformType[video::EMT_COUNT+2];
		video::E_SHADER_CONSTANT_TYPE nastyUniformType[video::EMT_COUNT+2];
		video::E_SHADER_CONSTANT_TYPE lightIDUniformType[video::EMT_COUNT+2];
	public:
		SimpleCallBack(Renderer* _renderer) : renderer(_renderer), currentMat(video::EMT_SOLID)
		{
			std::fill(mvpUniformLocation, reinterpret_cast<int32_t*>(mvpUniformType), -1);
		}

		virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::vector<video::SConstantLocationNamePair>& constants)
		{
			for (size_t i=0; i<constants.size(); i++)
			{
				if (constants[i].name == "MVP")
				{
					mvpUniformLocation[materialType] = constants[i].location;
					mvpUniformType[materialType] = constants[i].type;
				}
				else if (constants[i].name == "NormalMatrix")
				{
					nUniformLocation[materialType] = constants[i].location;
					nUniformType[materialType] = constants[i].type;
				}
				else if (constants[i].name == "flipFaces")
				{
					flipFacesUniformLocation[materialType] = constants[i].location;
					flipFacesUniformType[materialType] = constants[i].type;
				}
				else if (constants[i].name == "color")
				{
					colorUniformLocation[materialType] = constants[i].location;
					colorUniformType[materialType] = constants[i].type;
				}
				else if (constants[i].name == "nasty")
				{
					nastyUniformLocation[materialType] = constants[i].location;
					nastyUniformType[materialType] = constants[i].type;
				}
				else if (constants[i].name == "lightID")
				{
					lightIDUniformLocation[materialType] = constants[i].location;
					lightIDUniformType[materialType] = constants[i].type;
				}
			}
		}

		virtual void OnSetMaterial(video::IMaterialRendererServices* services, const video::SGPUMaterial& material, const video::SGPUMaterial& lastMaterial)
		{
			currentMat = material.MaterialType;
			services->setShaderConstant(&material.AmbientColor, colorUniformLocation[currentMat], colorUniformType[currentMat], 1);
			services->setShaderConstant(&material.MaterialTypeParam, nastyUniformLocation[currentMat], nastyUniformType[currentMat], 1);
			services->setShaderConstant(&material.userData, lightIDUniformLocation[currentMat], lightIDUniformType[currentMat], 1);
		}

		virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
		{
			auto world = services->getVideoDriver()->getTransform(video::E4X3TS_WORLD);
			if (flipFacesUniformLocation[currentMat]>=0)
			{
				auto tmp = world.getPseudoDeterminant();
				if (renderer->isRightHanded())
					tmp = -tmp;
				services->setShaderConstant(tmp.pointer, flipFacesUniformLocation[currentMat], flipFacesUniformType[currentMat], 1);
			}
			if (nUniformLocation[currentMat]>=0)
			{
				float tmp[9];
				world.getSub3x3InverseTransposePacked(tmp);
				services->setShaderConstant(tmp, nUniformLocation[currentMat], nUniformType[currentMat], 1);
			}
			if (mvpUniformLocation[currentMat]>=0)
				services->setShaderConstant(services->getVideoDriver()->getTransform(video::EPTS_PROJ_VIEW_WORLD).pointer(), mvpUniformLocation[currentMat], mvpUniformType[currentMat], 1);
		}

		virtual void OnUnsetMaterial() {}
};

constexpr uint32_t UNFLEXIBLE_MAX_SAMPLES_TODO_REMOVE = 1024u*1024u;


Renderer::Renderer(IVideoDriver* _driver, IAssetManager* _assetManager, ISceneManager* _smgr) :
		m_driver(_driver), m_smgr(_smgr), m_assetManager(_assetManager),
		nonInstanced(static_cast<E_MATERIAL_TYPE>(-1)), m_raygenProgram(0u), m_compostProgram(0u),
		m_rrManager(ext::RadeonRays::Manager::create(m_driver)), m_rightHanded(false),
		m_depth(), m_albedo(), m_normals(), m_lightIndex(), m_accumulation(), m_tonemapOutput(),
		m_colorBuffer(nullptr), m_gbuffer(nullptr), tmpTonemapBuffer(nullptr),
		m_maxSamples(0u), m_workGroupCount{0u,0u}, m_samplesPerDispatch(0u), m_samplesComputed(0u), m_rayCount(0u), m_framesDone(0u),
		m_rayBuffer(), m_intersectionBuffer(), m_rayCountBuffer(),
		m_rayBufferAsRR(nullptr,nullptr), m_intersectionBufferAsRR(nullptr,nullptr), m_rayCountBufferAsRR(nullptr,nullptr),
		nodes(), sceneBound(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX), rrInstances(),
		m_lightCount(0u)
{
	SimpleCallBack* cb = new SimpleCallBack(this);
	nonInstanced = (E_MATERIAL_TYPE)m_driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../mesh.vert",
		"", "", "", //! No Geometry or Tessellation Shaders
		"../mesh.frag",
		3, EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
		cb, //! Our Shader Callback
		0); //! No custom user data
	cb->drop();

	auto includes = m_rrManager->getRadeonRaysGLSLIncludes();
	m_raygenProgram = createComputeShader(
		raygenShaderExtensions+"#define MAX_SAMPLES "+std::to_string(UNFLEXIBLE_MAX_SAMPLES_TODO_REMOVE)+"\n"+
		//"irr/builtin/glsl/ext/RadeonRays/"
		includes->getBuiltinInclude("ray.glsl")+
		lightStruct+
		raygenShader
	);
	m_compostProgram = createComputeShader(
		compostShaderExtensions+
		//"irr/builtin/glsl/ext/RadeonRays/"
		includes->getBuiltinInclude("ray.glsl") +
		lightStruct+
		compostShader
	);

	// TODO: Upgrade to Icosphere!
	{
		auto tmpSphereMesh = m_assetManager->getGeometryCreator()->createSphereMesh(1.f, 16u, 16u);
		auto mb = tmpSphereMesh->getMeshBuffer(0u);

		uint32_t triangleCount = 0u;
		asset::IMeshManipulator::getPolyCount(triangleCount, mb);
		m_precomputedGeodesic.resize(triangleCount);
		for (auto triID=0u; triID<triangleCount; triID++)
		{
			auto triangle = asset::IMeshManipulator::getTriangleIndices(mb,triID);
			std::swap(triangle[1], triangle[2]); // make the sphere face inwards
			for (auto k=0u; k<3u; k++)
				m_precomputedGeodesic[triID][k] = mb->getPosition(triangle[k]);
		}
	}
}

Renderer::~Renderer()
{
	if (m_raygenProgram)
		COpenGLExtensionHandler::extGlDeleteProgram(m_raygenProgram);
	if (m_compostProgram)
		COpenGLExtensionHandler::extGlDeleteProgram(m_compostProgram);
}


void Renderer::init(const SAssetBundle& meshes,
					bool isCameraRightHanded,
					core::smart_refctd_ptr<ICPUBuffer>&& sampleSequence,
					uint32_t rayBufferSize)
{
	deinit();

	m_rightHanded = isCameraRightHanded;

	core::vector<SLight> lights;
	core::vector<uint32_t> lightCDF;
	core::vector<core::vectorSIMDf> lightRadiances;
	auto& lightPDF = reinterpret_cast<core::vector<float>&>(lightCDF); // save on memory

	const ext::MitsubaLoader::CGlobalMitsubaMetadata* globalMeta = nullptr;
	{
		auto contents = meshes.getContents();
		for (auto cpuit = contents.first; cpuit!=contents.second; cpuit++)
		{
			auto cpumesh = static_cast<asset::ICPUMesh*>(cpuit->get());

			auto* meta = cpumesh->getMetadata();

			assert(meta && core::strcmpi(meta->getLoaderName(), ext::MitsubaLoader::IMitsubaMetadata::LoaderName) == 0);
			const auto* meshmeta = static_cast<const ext::MitsubaLoader::IMeshMetadata*>(meta);
			globalMeta = meshmeta->globalMetadata.get();

			//! TODO: fix
			const auto shapeType = meshmeta->getShapeType()==ext::MitsubaLoader::CElementShape::Type::INSTANCE ? ext::MitsubaLoader::CElementShape::Type::SERIALIZED:meshmeta->getShapeType();

			const auto& instances = meshmeta->getInstances();

			auto meshBufferCount = cpumesh->getMeshBufferCount();
			for (auto i=0u; i<meshBufferCount; i++)
			{
				// TODO: get rid of `getMeshBuffer` and `getMeshBufferCount`, just return a range as `getMeshBuffers`
				auto cpumb = cpumesh->getMeshBuffer(i);
				m_rrManager->makeRRShapes(rrShapeCache, &cpumb, (&cpumb)+1);
			}
			
			for (auto instance : instances)
			{
				auto cpumesh = static_cast<ICPUMesh*>(cpuit->get());
				if (instance.emitter.type != ext::MitsubaLoader::CElementEmitter::Type::INVALID)
				{
					assert(instance.emitter.type==ext::MitsubaLoader::CElementEmitter::Type::AREA);

					uint32_t totalTriangleCount = 0u;
					asset::IMeshManipulator::getPolyCount(totalTriangleCount, cpumesh);

					SLight light;
					light.setFactor(instance.emitter.area.radiance);
					light.analytical = SLight::CachedTransform(instance.tform);

					bool bail = false;
					switch (shapeType)
					{
						case ext::MitsubaLoader::CElementShape::Type::SPHERE:
							light.type = SLight::ET_ELLIPSOID;
							light.analytical.transformCofactors = -light.analytical.transformCofactors;
							break;
						case ext::MitsubaLoader::CElementShape::Type::CYLINDER:
							_IRR_FALLTHROUGH;
						case ext::MitsubaLoader::CElementShape::Type::DISK:
							_IRR_FALLTHROUGH;
						case ext::MitsubaLoader::CElementShape::Type::RECTANGLE:
							_IRR_FALLTHROUGH;
						case ext::MitsubaLoader::CElementShape::Type::CUBE:
							_IRR_FALLTHROUGH;
						case ext::MitsubaLoader::CElementShape::Type::OBJ:
							_IRR_FALLTHROUGH;
						case ext::MitsubaLoader::CElementShape::Type::PLY:
							_IRR_FALLTHROUGH;
						case ext::MitsubaLoader::CElementShape::Type::SERIALIZED:
							light.type = SLight::ET_TRIANGLE;
							if (!totalTriangleCount)
								bail = true;
							break;
						default:
						#ifdef _DEBUG
							assert(false);
						#endif
							bail = true;
							break;
					}
					if (bail)
						continue;

					auto addLight = [&instance,&cpumesh,&lightPDF,&lights,&lightRadiances](auto& newLight,float approxArea) -> void
					{
						float weight = newLight.computeFlux(approxArea) * instance.emitter.area.samplingWeight;
						if (weight <= FLT_MIN)
							return;

						lightPDF.push_back(weight);
						lights.push_back(newLight);
						lightRadiances.push_back(newLight.strengthFactor);
					};
					auto areaFromTriangulationAndMakeMeshLight = [&]() -> float
					{
						double totalSurfaceArea = 0.0;

						uint32_t runningTriangleCount = 0u;
						for (auto i=0u; i<meshBufferCount; i++)
						{
							auto cpumb = cpumesh->getMeshBuffer(i);
							reinterpret_cast<uint32_t&>(cpumb->getMaterial().userData) = lights.size();

							uint32_t triangleCount = 0u;
							asset::IMeshManipulator::getPolyCount(triangleCount,cpumb);
							for (auto triID=0u; triID<triangleCount; triID++)
							{
								auto triangle = asset::IMeshManipulator::getTriangleIndices(cpumb,triID);

								core::vectorSIMDf v[3];
								for (auto k=0u; k<3u; k++)
									v[k] = cpumb->getPosition(triangle[k]);

								float triangleArea = NAN;
								auto triLight = SLight::createFromTriangle(instance.emitter.area.radiance, light.analytical, v, &triangleArea);
								if (light.type==SLight::ET_TRIANGLE)
									addLight(triLight,triangleArea);
								else
									totalSurfaceArea += triangleArea;
							}
						}
						return totalSurfaceArea;
					};

					auto totalArea = areaFromTriangulationAndMakeMeshLight();
					if (light.type!=SLight::ET_TRIANGLE)
						addLight(light,totalArea);
				}
				else // no emissive
				{
					for (auto i=0u; i<meshBufferCount; i++)
					{
						auto cpumb = cpumesh->getMeshBuffer(i);
						reinterpret_cast<uint32_t&>(cpumb->getMaterial().userData) = 0xdeadbeefu;
					}
				}
			}
		}

		auto gpumeshes = m_driver->getGPUObjectsFromAssets<ICPUMesh>(contents.first, contents.second);
		auto cpuit = contents.first;
		for (auto gpuit = gpumeshes->begin(); gpuit!=gpumeshes->end(); gpuit++,cpuit++)
		{
			auto* meta = cpuit->get()->getMetadata();

			assert(meta && core::strcmpi(meta->getLoaderName(),ext::MitsubaLoader::IMitsubaMetadata::LoaderName) == 0);
			const auto* meshmeta = static_cast<const ext::MitsubaLoader::IMeshMetadata*>(meta);

			const auto& instances = meshmeta->getInstances();

			const auto& gpumesh = *gpuit;
			for (auto i=0u; i<gpumesh->getMeshBufferCount(); i++)
				gpumesh->getMeshBuffer(i)->getMaterial().MaterialType = nonInstanced;

			for (auto instance : instances)
			{
				auto node = core::smart_refctd_ptr<IMeshSceneNode>(m_smgr->addMeshSceneNode(core::smart_refctd_ptr(gpumesh)));
				node->setRelativeTransformationMatrix(instance.tform.getAsRetardedIrrlichtMatrix());
				node->updateAbsolutePosition();
				sceneBound.addInternalBox(node->getTransformedBoundingBox());

				nodes.push_back(std::move(node));
			}
		}
		core::vector<int32_t> ids(nodes.size());
		std::iota(ids.begin(), ids.end(), 0);
		auto nodesBegin = &nodes.data()->get();
		m_rrManager->makeRRInstances(rrInstances, rrShapeCache, m_assetManager, nodesBegin, nodesBegin+nodes.size(), ids.data());
		m_rrManager->attachInstances(rrInstances.begin(), rrInstances.end());
	}

	auto renderSize = m_driver->getScreenSize();
	if (globalMeta)
	{
		constantClearColor.set(0.f, 0.f, 0.f, 1.f);
		float constantCombinedWeight = 0.f;
		for (auto emitter : globalMeta->emitters)
		{
			SLight light;

			float weight = 0.f;
			switch (emitter.type)
			{
				case ext::MitsubaLoader::CElementEmitter::Type::CONSTANT:
					constantClearColor += emitter.constant.radiance;
					constantCombinedWeight += emitter.constant.samplingWeight;
					break;
				case ext::MitsubaLoader::CElementEmitter::Type::INVALID:
					break;
				default:
				#ifdef _DEBUG
					assert(false);
				#endif
					//weight = emitter..samplingWeight;
					//light.type = SLight::ET_;
					//light.setFactor(emitter..radiance);
					break;
			}
			if (weight==0.f)
				continue;
			
			weight *= light.computeFlux(NAN);
			if (weight <= FLT_MIN)
				continue;

			lightPDF.push_back(weight);
			lights.push_back(light);
		}
		// add constant light
		if (constantCombinedWeight>FLT_MIN)
		{
			core::matrix3x4SIMD tform;
			tform.setScale(core::vectorSIMDf().set(sceneBound.getExtent())*core::sqrt(3.f/4.f));
			tform.setTranslation(core::vectorSIMDf().set(sceneBound.getCenter()));
			SLight::CachedTransform cachedT(tform);

			const float areaOfSphere = 4.f * core::PI<float>();

			auto startIx = lights.size();
			float triangulationArea = 0.f;
			for (const auto& tri : m_precomputedGeodesic)
			{
				// make light, correct radiance parameter for domain of integration
				float area = NAN;
				auto triLight = SLight::createFromTriangle(constantClearColor,cachedT,&tri[0],&area);
				// compute flux, abuse of notation on area parameter, we've got the wrong units on strength factor so need to cancel
				float flux = triLight.computeFlux(1.f);
				// add to light list
				float weight = constantCombinedWeight*flux;
				if (weight>FLT_MIN)
				{
					// needs to be weighted be contribution to sphere area
					lightPDF.push_back(weight*area);
					lights.push_back(triLight);
					lightRadiances.push_back(constantClearColor);
				}
				triangulationArea += area;
			}

			for (auto i=startIx; i<lights.size(); i++)
				lightPDF[i] *= areaOfSphere/triangulationArea;
		}

		if (globalMeta->sensors.size())
		{
			const auto& sensor = globalMeta->sensors.front();
			const auto& film = sensor.film;
			renderSize.set(film.cropWidth,film.cropHeight);
		}
	}

	//! TODO: move out into a function `finalizeLights`
	if (lights.size())
	{
		double weightSum = 0.0;
		for (auto i=0u; i<lightPDF.size(); i++)
			weightSum += lightPDF[i];
		assert(weightSum>FLT_MIN);

		constexpr double UINT_MAX_DOUBLE = double(0x1ull<<32ull);

		double weightSumRcp = UINT_MAX_DOUBLE/weightSum;
		double partialSum = 0.0;
		for (auto i=0u; i<lightCDF.size(); i++)
		{
			uint32_t prevCDF = i ? lightCDF[i-1u]:0u;

			double pdf = lightPDF[i];
			partialSum += pdf;

			double inv_prob = NAN;
			double exactCDF = partialSum*weightSumRcp+double(FLT_MIN);
			if (exactCDF<UINT_MAX_DOUBLE)
			{
				lightCDF[i] = static_cast<uint32_t>(exactCDF);
				inv_prob = UINT_MAX_DOUBLE/(lightCDF[i]-prevCDF);
			}
			else
			{
				assert(exactCDF<UINT_MAX_DOUBLE+1.0);
				lightCDF[i] = 0xdeadbeefu;
				inv_prob = 1.0/(1.0-double(prevCDF)/UINT_MAX_DOUBLE);
			}
			lights[i].setFactor(core::vectorSIMDf(lights[i].strengthFactor)*inv_prob);
		}
		lightCDF.back() = 0xdeadbeefu;

		m_lightCount = lights.size();
		m_lightCDFBuffer = core::smart_refctd_ptr<IGPUBuffer>(m_driver->createFilledDeviceLocalGPUBufferOnDedMem(lightCDF.size()*sizeof(uint32_t),lightCDF.data()),core::dont_grab);
		m_lightBuffer = core::smart_refctd_ptr<IGPUBuffer>(m_driver->createFilledDeviceLocalGPUBufferOnDedMem(lights.size()*sizeof(SLight),lights.data()),core::dont_grab);
		m_lightRadianceBuffer = core::smart_refctd_ptr<IGPUBuffer>(m_driver->createFilledDeviceLocalGPUBufferOnDedMem(lightRadiances.size()*sizeof(core::vectorSIMDf),lightRadiances.data()),core::dont_grab);
	}

	//! set up GPU sampler 
	{
		m_maxSamples = sampleSequence->getSize()/(sizeof(uint32_t)*MaxDimensions);
		assert(m_maxSamples==UNFLEXIBLE_MAX_SAMPLES_TODO_REMOVE);

		// upload sequence to GPU
		auto gpubuf = m_driver->createFilledDeviceLocalGPUBufferOnDedMem(sampleSequence->getSize(), sampleSequence->getPointer());
		m_sampleSequence = core::smart_refctd_ptr<ITextureBufferObject>(m_driver->addTextureBufferObject(gpubuf, ITextureBufferObject::ETBOF_RG32UI), core::dont_grab);
		gpubuf->drop();

		// create scramble texture
		uint32_t res[3] = {renderSize.Width,renderSize.Height,1u};
		m_scrambleTexture = m_driver->createGPUTexture(ITexture::ETT_2D, &res[0], 1u, EF_R32_UINT);
		{
			core::vector<uint32_t> random(res[0]*res[1]);
			// generate
			{
				core::RandomSampler rng(0xbadc0ffeu);
				for (auto& pixel : random)
					pixel = rng.nextSample();
			}
			// upload
			uint32_t _min[3] = {0u,0u,0u};
			m_scrambleTexture->updateSubRegion(EF_R32_UINT, random.data(), &_min[0], &res[0]);
		}
	}

	m_depth = m_driver->createGPUTexture(ITexture::ETT_2D, &renderSize.Width, 1, EF_D32_SFLOAT);
	m_albedo = m_driver->createGPUTexture(ITexture::ETT_2D, &renderSize.Width, 1, EF_R8G8B8_SRGB);
	m_normals = m_driver->createGPUTexture(ITexture::ETT_2D, &renderSize.Width, 1, EF_R16G16_SNORM);
	m_lightIndex = m_driver->createGPUTexture(ITexture::ETT_2D, &renderSize.Width, 1, EF_R32_UINT);

	m_accumulation = m_driver->createGPUTexture(ITexture::ETT_2D, &renderSize.Width, 1, EF_R32G32B32A32_SFLOAT);
	m_tonemapOutput = m_driver->createGPUTexture(ITexture::ETT_2D, &renderSize.Width, 1, EF_R8G8B8_SRGB);

	m_colorBuffer = m_driver->addFrameBuffer();
	m_colorBuffer->attach(EFAP_COLOR_ATTACHMENT0, m_tonemapOutput.get());

	tmpTonemapBuffer = m_driver->addFrameBuffer();
	tmpTonemapBuffer->attach(EFAP_COLOR_ATTACHMENT0, m_accumulation.get());

	m_gbuffer = m_driver->addFrameBuffer();
	m_gbuffer->attach(EFAP_DEPTH_ATTACHMENT, m_depth.get());
	m_gbuffer->attach(EFAP_COLOR_ATTACHMENT0, m_albedo.get());
	m_gbuffer->attach(EFAP_COLOR_ATTACHMENT1, m_normals.get());
	m_gbuffer->attach(EFAP_COLOR_ATTACHMENT2, m_lightIndex.get());

	//
#define WORK_GROUP_DIM 16u
	m_workGroupCount[0] = (renderSize.Width+WORK_GROUP_DIM-1)/WORK_GROUP_DIM;
	m_workGroupCount[1] = (renderSize.Height+WORK_GROUP_DIM-1)/WORK_GROUP_DIM;
	uint32_t pixelCount = renderSize.Width*renderSize.Height;

	auto raygenBufferSize = static_cast<size_t>(pixelCount)*sizeof(::RadeonRays::ray);
	assert(raygenBufferSize<=rayBufferSize);
	auto shadowBufferSize = static_cast<size_t>(pixelCount)*sizeof(int32_t);
	assert(shadowBufferSize<=rayBufferSize);
	m_samplesPerDispatch = rayBufferSize/(raygenBufferSize+shadowBufferSize);
	assert(m_samplesPerDispatch >= 1u);
	printf("Using %d samples\n", m_samplesPerDispatch);

	raygenBufferSize *= m_samplesPerDispatch;
	m_rayBuffer = core::smart_refctd_ptr<IGPUBuffer>(m_driver->createDeviceLocalGPUBufferOnDedMem(raygenBufferSize), core::dont_grab);

	shadowBufferSize *= m_samplesPerDispatch;
	m_intersectionBuffer = core::smart_refctd_ptr<IGPUBuffer>(m_driver->createDeviceLocalGPUBufferOnDedMem(shadowBufferSize),core::dont_grab);

	m_rayCount = m_samplesPerDispatch*pixelCount;
	m_rayCountBuffer = core::smart_refctd_ptr<IGPUBuffer>(m_driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(uint32_t),&m_rayCount), core::dont_grab);

	m_rayBufferAsRR = m_rrManager->linkBuffer(m_rayBuffer.get(), CL_MEM_READ_WRITE);
	// TODO: clear hit buffer to -1 before usage
	m_intersectionBufferAsRR = m_rrManager->linkBuffer(m_intersectionBuffer.get(), CL_MEM_READ_WRITE);
	m_rayCountBufferAsRR = m_rrManager->linkBuffer(m_rayCountBuffer.get(), CL_MEM_READ_ONLY);

	const cl_mem clObjects[] = { m_rayCountBufferAsRR.second };
	auto objCount = sizeof(clObjects)/sizeof(cl_mem);
	clEnqueueAcquireGLObjects(m_rrManager->getCLCommandQueue(), objCount, clObjects, 0u, nullptr, nullptr);
}


void Renderer::deinit()
{
	auto commandQueue = m_rrManager->getCLCommandQueue();
	clFinish(commandQueue);

	glFinish();

	// create a screenshot
	if (m_tonemapOutput)
		ext::ScreenShot::dirtyCPUStallingScreenshot(m_driver, m_assetManager, "screenshot.png", m_tonemapOutput.get());

	// release OpenCL objects and wait for OpenCL to finish
	const cl_mem clObjects[] = { m_rayCountBufferAsRR.second };
	auto objCount = sizeof(clObjects) / sizeof(cl_mem);
	clEnqueueReleaseGLObjects(commandQueue, objCount, clObjects, 1u, nullptr, nullptr);
	clFlush(commandQueue);
	clFinish(commandQueue);

	if (m_colorBuffer)
	{
		m_driver->removeFrameBuffer(m_colorBuffer);
		m_colorBuffer = nullptr;
	}
	if (m_gbuffer)
	{
		m_driver->removeFrameBuffer(m_gbuffer);
		m_gbuffer = nullptr;
	}
	if (tmpTonemapBuffer)
	{
		m_driver->removeFrameBuffer(tmpTonemapBuffer);
		tmpTonemapBuffer = nullptr;
	}

	m_depth = m_albedo = m_normals = m_lightIndex = m_accumulation = m_tonemapOutput = nullptr;

	if (m_rayBufferAsRR.first)
	{
		m_rrManager->deleteRRBuffer(m_rayBufferAsRR.first);
		m_rayBufferAsRR = {nullptr,nullptr};
	}
	if (m_intersectionBufferAsRR.first)
	{
		m_rrManager->deleteRRBuffer(m_intersectionBufferAsRR.first);
		m_intersectionBufferAsRR = {nullptr,nullptr};
	}
	if (m_rayCountBufferAsRR.first)
	{
		m_rrManager->deleteRRBuffer(m_rayCountBufferAsRR.first);
		m_rayCountBufferAsRR = {nullptr,nullptr};
	}
	m_rayBuffer = m_intersectionBuffer = m_rayCountBuffer = nullptr;
	m_samplesPerDispatch = 0u;
	m_samplesComputed = 0u;
	m_rayCount = 0u;
	m_framesDone = 0u;
	m_workGroupCount[0] = m_workGroupCount[1] = 0u;
	m_maxSamples = 0u;

	for (auto& node : nodes)
		node->remove();
	nodes.clear();
	sceneBound = core::aabbox3df(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX);
	m_rrManager->deleteShapes(rrShapeCache.begin(),rrShapeCache.end());
	rrShapeCache.clear();
	m_rrManager->detachInstances(rrInstances.begin(),rrInstances.end());
	m_rrManager->deleteInstances(rrInstances.begin(),rrInstances.end());
	rrInstances.clear();

	constantClearColor.set(0.f,0.f,0.f,1.f);
	m_lightCount = 0u;
	m_lightCDFBuffer = m_lightBuffer = m_lightRadianceBuffer = nullptr;

	// start deleting objects
	m_sampleSequence = nullptr;
	m_scrambleTexture = nullptr;

	m_rightHanded = false;
}


void Renderer::render()
{
	m_driver->setRenderTarget(m_gbuffer);
	{ // clear
		m_driver->clearZBuffer();
		float zero[4] = { 0.f,0.f,0.f,0.f };
		m_driver->clearColorBuffer(EFAP_COLOR_ATTACHMENT0, zero);
		m_driver->clearColorBuffer(EFAP_COLOR_ATTACHMENT1, zero);
		uint32_t clearLightID[4] = {((constantClearColor>core::vectorSIMDf(0.f))&core::vectorSIMDu32(~0u,~0u,~0u,0u)).any() ? (m_lightCount-1u):0xdeadbeefu,0,0,0};
		m_driver->clearColorBuffer(EFAP_COLOR_ATTACHMENT2, clearLightID);
	}

	auto camera = m_smgr->getActiveCamera();
	auto prevViewProj = camera->getConcatenatedMatrix();

	//! This animates (moves) the camera and sets the transforms
	//! Also draws the meshbuffer
	m_smgr->drawAll();

	auto currentViewProj = camera->getConcatenatedMatrix();
	if (!core::equals(prevViewProj,currentViewProj,core::ROUNDING_ERROR<core::matrix4SIMD>()*100.0))
	{
		m_samplesComputed = 0u;
		m_framesDone = 0u;
	}

	auto rSize = m_depth->getSize();

	// generate rays
	{
		GLint prevProgram;
		glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);

		STextureSamplingParams params;
		params.MaxFilter = ETFT_NEAREST_NEARESTMIP;
		params.MinFilter = ETFT_NEAREST_NEARESTMIP;
		params.UseMipmaps = 0;

		const COpenGLDriver::SAuxContext* foundConst = static_cast<COpenGLDriver*>(m_driver)->getThreadContext();
		COpenGLDriver::SAuxContext* found = const_cast<COpenGLDriver::SAuxContext*>(foundConst);
		found->setActiveTexture(0, core::smart_refctd_ptr(m_depth), params);
		found->setActiveTexture(1, core::smart_refctd_ptr(m_normals), params);
		found->setActiveTexture(2, core::smart_refctd_ptr(m_albedo), params);
		found->setActiveTexture(3, core::smart_refctd_ptr(m_sampleSequence), params);
		found->setActiveTexture(4, core::smart_refctd_ptr(m_scrambleTexture), params);


		COpenGLExtensionHandler::extGlUseProgram(m_raygenProgram);

		const COpenGLBuffer* buffers[] = { static_cast<const COpenGLBuffer*>(m_rayBuffer.get()),static_cast<const COpenGLBuffer*>(m_lightCDFBuffer.get()),static_cast<const COpenGLBuffer*>(m_lightBuffer.get()) };
		ptrdiff_t offsets[] = { 0,0,0 };
		ptrdiff_t sizes[] = { m_rayBuffer->getSize(),m_lightCDFBuffer->getSize(),m_lightBuffer->getSize() };
		found->setActiveSSBO(0, sizeof(offsets)/sizeof(ptrdiff_t), buffers, offsets, sizes);

		{
			auto camPos = core::vectorSIMDf().set(camera->getAbsolutePosition());
			COpenGLExtensionHandler::pGlProgramUniform3fv(m_raygenProgram, 0, 1, camPos.pointer);

			float uDepthLinearizationConstant;
			{
				auto projMat = camera->getProjectionMatrix();
				auto* row = projMat.rows;
				uDepthLinearizationConstant = -row[3][2]/(row[3][2]-row[2][2]);
			}
			COpenGLExtensionHandler::pGlProgramUniform1fv(m_raygenProgram, 1, 1, &uDepthLinearizationConstant);

			auto frustum = camera->getViewFrustum();
			core::matrix4SIMD uFrustumCorners;
			uFrustumCorners.rows[1] = frustum->getFarLeftDown();
			uFrustumCorners.rows[0] = frustum->getFarRightDown()-uFrustumCorners.rows[1];
			uFrustumCorners.rows[1] -= camPos;
			uFrustumCorners.rows[3] = frustum->getFarLeftUp();
			uFrustumCorners.rows[2] = frustum->getFarRightUp()-uFrustumCorners.rows[3];
			uFrustumCorners.rows[3] -= camPos;
			COpenGLExtensionHandler::pGlProgramUniformMatrix4fv(m_raygenProgram, 2, 1, false, uFrustumCorners.pointer()); // important to say no to transpose

			COpenGLExtensionHandler::pGlProgramUniform2uiv(m_raygenProgram, 3, 1, rSize);

			uint32_t uSamples_ImageWidthSamples[2] = {m_samplesPerDispatch,rSize[0]*m_samplesPerDispatch};
			COpenGLExtensionHandler::pGlProgramUniform2uiv(m_raygenProgram, 4, 1, uSamples_ImageWidthSamples);

			COpenGLExtensionHandler::pGlProgramUniform1uiv(m_raygenProgram, 5, 1, &m_samplesComputed);
			m_samplesComputed += m_samplesPerDispatch;

			float uImageSize2Rcp[4] = {1.f/static_cast<float>(rSize[0]),1.f/static_cast<float>(rSize[1]),0.5f/static_cast<float>(rSize[0]),0.5f/static_cast<float>(rSize[1])};
			COpenGLExtensionHandler::pGlProgramUniform4fv(m_raygenProgram, 6, 1, uImageSize2Rcp);
		}

		COpenGLExtensionHandler::pGlDispatchCompute(m_workGroupCount[0], m_workGroupCount[1], 1);

		COpenGLExtensionHandler::extGlUseProgram(prevProgram);
		
		// probably wise to flush all caches
		COpenGLExtensionHandler::pGlMemoryBarrier(GL_ALL_BARRIER_BITS);
	}

	// do radeon rays
	m_rrManager->update(rrInstances);
	if (m_rrManager->hasImplicitCL2GLSync())
		glFlush();
	else
		glFinish();

	auto commandQueue = m_rrManager->getCLCommandQueue();
	{
		const cl_mem clObjects[] = {m_rayBufferAsRR.second,m_intersectionBufferAsRR.second};
		auto objCount = sizeof(clObjects)/sizeof(cl_mem);

		cl_event acquired = nullptr;
		clEnqueueAcquireGLObjects(commandQueue,objCount,clObjects,0u,nullptr,&acquired);

		clEnqueueWaitForEvents(commandQueue,1u,&acquired);
		m_rrManager->getRadeonRaysAPI()->QueryOcclusion(m_rayBufferAsRR.first,m_rayCountBufferAsRR.first,m_rayCount,m_intersectionBufferAsRR.first,nullptr,nullptr);
		cl_event raycastDone = nullptr;
		clEnqueueMarker(commandQueue,&raycastDone);

		if (m_rrManager->hasImplicitCL2GLSync())
		{
			clEnqueueReleaseGLObjects(commandQueue, objCount, clObjects, 1u, &raycastDone, nullptr);
			clFlush(commandQueue);
		}
		else
		{
			cl_event released;
			clEnqueueReleaseGLObjects(commandQueue, objCount, clObjects, 1u, &raycastDone, &released);
			clFlush(commandQueue);
			clWaitForEvents(1u, &released);
		}
	}

	// use raycast results
	{
		GLint prevProgram;
		glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);

		STextureSamplingParams params;
		params.MaxFilter = ETFT_NEAREST_NEARESTMIP;
		params.MinFilter = ETFT_NEAREST_NEARESTMIP;
		params.UseMipmaps = 0;

		const COpenGLDriver::SAuxContext* foundConst = static_cast<COpenGLDriver*>(m_driver)->getThreadContext();
		COpenGLDriver::SAuxContext* found = const_cast<COpenGLDriver::SAuxContext*>(foundConst);
		found->setActiveTexture(0, core::smart_refctd_ptr(m_lightIndex), params);
		
		COpenGLExtensionHandler::extGlBindImageTexture(0u,static_cast<COpenGLFilterableTexture*>(m_accumulation.get())->getOpenGLName(),0,false,0,GL_READ_WRITE,GL_RGBA32F);

		COpenGLExtensionHandler::extGlUseProgram(m_compostProgram);

		const COpenGLBuffer* buffers[] ={	static_cast<const COpenGLBuffer*>(m_rayBuffer.get()),
											static_cast<const COpenGLBuffer*>(m_intersectionBuffer.get()),
											static_cast<const COpenGLBuffer*>(m_lightRadianceBuffer.get())
										};
		ptrdiff_t offsets[] = { 0,0,0 };
		ptrdiff_t sizes[] = { m_rayBuffer->getSize(),m_intersectionBuffer->getSize(),m_lightRadianceBuffer->getSize() };
		found->setActiveSSBO(0, 3, buffers, offsets, sizes);

		{
			COpenGLExtensionHandler::pGlProgramUniform2uiv(m_compostProgram, 0, 1, rSize);
			
			uint32_t uSamples_ImageWidthSamples[2] = {m_samplesPerDispatch,rSize[0]*m_samplesPerDispatch};
			COpenGLExtensionHandler::pGlProgramUniform2uiv(m_compostProgram, 1, 1, uSamples_ImageWidthSamples);

			m_framesDone++;
			float uRcpFramesDone = 1.0/double(m_framesDone);
			COpenGLExtensionHandler::pGlProgramUniform1fv(m_compostProgram, 2, 1, &uRcpFramesDone);
		}

		COpenGLExtensionHandler::pGlDispatchCompute(m_workGroupCount[0], m_workGroupCount[1], 1);
		
		COpenGLExtensionHandler::extGlBindImageTexture(0u, 0u, 0, false, 0, GL_INVALID_ENUM, GL_INVALID_ENUM);

		COpenGLExtensionHandler::extGlUseProgram(prevProgram);

		COpenGLExtensionHandler::pGlMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT);
	}

	// TODO: tonemap properly
	{
		auto oldVP = m_driver->getViewPort();
		m_driver->blitRenderTargets(tmpTonemapBuffer, m_colorBuffer, false, false, {}, {}, true);
		m_driver->setViewPort(oldVP);
	}
}
#endif