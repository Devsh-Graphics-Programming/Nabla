#include <numeric>

#include "ExtraCrap.h"

#include "../../ext/ScreenShot/ScreenShot.h"

#include "../../ext/MitsubaLoader/CMitsubaLoader.h"

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
layout(location = 5) uniform uint uTotalSamplesComputed;
layout(location = 6) uniform vec4 uImageSize2Rcp;

// image views
layout(binding = 0) uniform sampler2D depthbuf;
layout(binding = 1) uniform sampler2D normalbuf;
layout(binding = 2) uniform sampler2D albedobuf;

// SSBOs
layout(binding = 0, std430) restrict writeonly buffer Rays
{
	RadeonRays_ray rays[];
};

layout(binding = 1, std430) restrict readonly buffer CumulativeLightPDF
{
	uint lightCDF[];
};

#define SLight_ET_CONSTANT	0u
#define SLight_ET_CUBE		1u
#define SLight_ET_ELLIPSOID	2u
#define SLight_ET_CYLINDER	3u
#define SLight_ET_RECTANGLE	4u
#define SLight_ET_DISK		5u
#define SLight_ET_TRIANGLE	6u
#define SLight_ET_COUNT		7u
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

uint wang_hash(uint seed)
{
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);
    return seed;
}

uint rand_xorshift(inout uint rng_state)
{
    // Xorshift algorithm from George Marsaglia's paper
    rng_state ^= (rng_state << 13);
    rng_state ^= (rng_state >> 17);
    rng_state ^= (rng_state << 5);
    return rng_state;
}

uint ugen_uniform_sample1(inout uint state)
{
	return rand_xorshift(state);
}
uvec2 ugen_uniform_sample2(inout uint state)
{
	return uvec2(rand_xorshift(state),rand_xorshift(state));
}

vec2 gen_uniform_sample2(inout uint state)
{
	return vec2(rand_xorshift(state),rand_xorshift(state))/vec2(~0u);
}

// https://orbit.dtu.dk/files/126824972/onb_frisvad_jgt2012_v2.pdf
mat2x3 frisvad(in vec3 n)
{
	const float a = 1.0/(1.0 + n.z);
	const float b = -n.x*n.y*a;
	return (n.z<-0.9999999) ? mat2x3(vec3(0.0,-1.0,0.0),vec3(-1.0,0.0,0.0)):mat2x3(vec3(1.0-n.x*n.x*a, b, -n.x),vec3(b, 1.0-n.y*n.y*a, -n.y));
}


uint lower_bound(in uint key)
{
	uint size = uint(lightCDF.length());
    uint low = 0u;

#define ITERATION \
	{\
        uint _half = size >> 1u; \
        uint other_half = size - _half; \
        uint probe = low + _half; \
        uint other_low = low + other_half; \
        uint v = lightCDF[probe]; \
        size = _half; \
        low = key>v ? other_low:low; \
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


vec3 light_sample(out vec3 incoming, inout uint randomState, inout float maxT, inout bool alive, in vec3 position, in vec3 normal)
{
	uint lightIDSample = ugen_uniform_sample1(randomState);
	vec2 lightSurfaceSample = gen_uniform_sample2(randomState);

	uint lightID = lower_bound(lightIDSample);

	SLight light = light[lightID];

#define SHADOW_RAY_LEN 0.5
	float factor; // 1.0/light_probability already baked into the light factor
	switch (SLight_extractType(light))
	{
		case SLight_ET_CUBE:
			{
				mat4x3 tform = light.transform;
				vec3 toCube = tform[3]-position;
				vec3 histogram = toCube.xxx+vec3(0.0,toCube.yy)+vec3(0.0,0.0,toCube.z);
				uint subFaceID = lightSurfaceSample.y>histogram.x ? (lightSurfaceSample.y>histogram.y ? 2u:1u):0u;

				float faceDP = toCube[subFaceID];
				toCube[subFaceID] -= sign(faceDP);
				float v = (lightSurfaceSample.y-histogram[subFaceID])*2.0/toCube[subFaceID]-1.0;

				uvec2 tanID[] = uvec2[](uvec2(1,2),uvec2(0,2),uvec2(0,1));
				toCube[tanID[subFaceID][0]] += lightSurfaceSample.x*2.0-1.0;
				toCube[tanID[subFaceID][1]] += v;
 
				incoming = toCube;
				float incomingInvLen = inversesqrt(dot(incoming,incoming));
				incoming *= incomingInvLen;

				maxT = SHADOW_RAY_LEN/incomingInvLen;

				factor = 12.0; // compensate for the domain of integration
				factor *= abs(faceDP)*incomingInvLen*incomingInvLen;
			}
			break;
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
				vec3 lightNormal = light.transformCofactors*pointOnSurface;

				factor *= max(dot(lightNormal,incoming),0.0)*incomingInvLen*incomingInvLen;
			}
			break;
		case SLight_ET_CYLINDER:
			{
				float equator = lightSurfaceSample.y*2.0*kPI;
				vec3 pointOnSurface = vec3(cos(equator),sin(equator),lightSurfaceSample.x);
	
				mat4x3 tform = light.transform;
				incoming = mat3(tform)*pointOnSurface+(tform[3]-position);
				float incomingInvLen = inversesqrt(dot(incoming,incoming));
				incoming *= incomingInvLen;

				maxT = SHADOW_RAY_LEN/incomingInvLen;

				factor = 2.0*kPI; // compensate for the domain of integration
				// don't normalize, length of the normal times determinant is very handy for differential area after a 3x3 matrix transform
				vec3 lightNormal = light.transformCofactors[0]*pointOnSurface.x+light.transformCofactors[1]*pointOnSurface.y;

				factor *= max(dot(lightNormal,incoming),0.0)*incomingInvLen*incomingInvLen;
			}
			break;
		case SLight_ET_RECTANGLE:
			{
				vec3 pointOnSurface = vec3(lightSurfaceSample*2.0-vec2(1.0),0.0000001); // TODO: FLT_MIN
	
				mat4x3 tform = light.transform;
				incoming = mat3(tform)*pointOnSurface+(tform[3]-position);
				float incomingInvLen = inversesqrt(dot(incoming,incoming));
				incoming *= incomingInvLen;

				maxT = SHADOW_RAY_LEN/incomingInvLen;

				factor = 4.0; // compensate for the domain of integration
				// don't normalize, length of the normal times determinant is very handy for differential area after a 3x3 matrix transform
				factor *= max(dot(-light.transformCofactors[2],incoming),0.0)*incomingInvLen*incomingInvLen;
			}
			break;
		case SLight_ET_DISK:
			{
				float equator = lightSurfaceSample.y*2.0*kPI;
				vec3 pointOnSurface = vec3(vec2(cos(equator),sin(equator))*sqrt(lightSurfaceSample.x),0.0000001); // TODO: FLT_MIN
	
				mat4x3 tform = light.transform;
				incoming = mat3(tform)*pointOnSurface+(tform[3]-position);
				float incomingInvLen = inversesqrt(dot(incoming,incoming));
				incoming *= incomingInvLen;

				maxT = SHADOW_RAY_LEN/incomingInvLen;

				factor = kPI; // compensate for the domain of integration
				// don't normalize, length of the normal times determinant is very handy for differential area after a 3x3 matrix transform
				factor *= max(dot(-light.transformCofactors[2],incoming),0.0)*incomingInvLen*incomingInvLen;
			}
			break;
		case SLight_ET_TRIANGLE:
			{
				mat3 vertices = transpose(mat3(light.transform));
				vec3 pointOnSurface = vertices[0];

				incoming = pointOnSurface-position;
				float incomingInvLen = inversesqrt(dot(incoming,incoming));
				incoming *= incomingInvLen;

				maxT = SHADOW_RAY_LEN/incomingInvLen;

				vec3 lightNormal = cross(vertices[1]-vertices[0],vertices[2]-vertices[0]);
				factor = 0.5*max(dot(lightNormal,incoming),0.0)*incomingInvLen*incomingInvLen;
			}
			break;
		default: // SLight_ET_CONSTANT:
			{
				float equator = lightSurfaceSample.y*2.0*kPI;
				vec3 pointOnSphere = vec3(vec2(cos(equator),sin(equator))*sqrt(1.0-lightSurfaceSample.x*lightSurfaceSample.x),lightSurfaceSample.x);
	
				mat2x3 tangents = frisvad(normal);
				incoming = mat3(tangents[0],tangents[1],normal)*pointOnSphere;

				factor = kPI*2.0; // compensate for the domain of integration
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

		uint randomState = wang_hash(uTotalSamplesComputed+outputID);

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
			alive = alive && lightCDF.length()!=0;
			if (alive)
				bsdf.rgb = light_sample(newray.direction,randomState,newray.maxT,alive,position,normal);
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
		// TODO: move the `div` to tonemapping shader
		color /= float(uSamples_ImageWidthSamples.x);

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
		video::E_MATERIAL_TYPE currentMat;

		int32_t mvpUniformLocation[video::EMT_COUNT+2];
		int32_t nUniformLocation[video::EMT_COUNT+2];
		int32_t colorUniformLocation[video::EMT_COUNT+2];
		int32_t nastyUniformLocation[video::EMT_COUNT+2];
		video::E_SHADER_CONSTANT_TYPE mvpUniformType[video::EMT_COUNT+2];
		video::E_SHADER_CONSTANT_TYPE nUniformType[video::EMT_COUNT+2];
		video::E_SHADER_CONSTANT_TYPE colorUniformType[video::EMT_COUNT+2];
		video::E_SHADER_CONSTANT_TYPE nastyUniformType[video::EMT_COUNT+2];
	public:
		SimpleCallBack() : currentMat(video::EMT_SOLID)
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
			}
		}

		virtual void OnSetMaterial(video::IMaterialRendererServices* services, const video::SGPUMaterial& material, const video::SGPUMaterial& lastMaterial)
		{
			currentMat = material.MaterialType;
			services->setShaderConstant(&material.AmbientColor, colorUniformLocation[currentMat], colorUniformType[currentMat], 1);
			services->setShaderConstant(&material.MaterialTypeParam, nastyUniformLocation[currentMat], nastyUniformType[currentMat], 1);
		}

		virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
		{
			if (nUniformLocation[currentMat]>=0)
			{
				float tmp[9];
				services->getVideoDriver()->getTransform(video::E4X3TS_WORLD).getSub3x3InverseTransposePacked(tmp);
				services->setShaderConstant(tmp, nUniformLocation[currentMat], nUniformType[currentMat], 1);
			}
			if (mvpUniformLocation[currentMat]>=0)
				services->setShaderConstant(services->getVideoDriver()->getTransform(video::EPTS_PROJ_VIEW_WORLD).pointer(), mvpUniformLocation[currentMat], mvpUniformType[currentMat], 1);
		}

		virtual void OnUnsetMaterial() {}
};


Renderer::Renderer(IVideoDriver* _driver, IAssetManager* _assetManager, ISceneManager* _smgr) :
		m_driver(_driver), m_smgr(_smgr), m_assetManager(_assetManager),
		nonInstanced(static_cast<E_MATERIAL_TYPE>(-1)), m_raygenProgram(0u), m_compostProgram(0u),
		m_rrManager(ext::RadeonRays::Manager::create(m_driver)),
		m_depth(), m_albedo(), m_normals(), m_accumulation(), m_tonemapOutput(),
		m_colorBuffer(nullptr), m_gbuffer(nullptr), tmpTonemapBuffer(nullptr),
		m_workGroupCount{0u,0u}, m_samplesPerDispatch(0u), m_totalSamplesComputed(0u), m_rayCount(0u), m_framesDone(0u),
		m_rayBuffer(), m_intersectionBuffer(), m_rayCountBuffer(),
		m_rayBufferAsRR(nullptr,nullptr), m_intersectionBufferAsRR(nullptr,nullptr), m_rayCountBufferAsRR(nullptr,nullptr),
		nodes(), sceneBound(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX), rrInstances()
{
	SimpleCallBack* cb = new SimpleCallBack();
	nonInstanced = (E_MATERIAL_TYPE)m_driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../mesh.vert",
		"", "", "", //! No Geometry or Tessellation Shaders
		"../mesh.frag",
		3, EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
		cb, //! Our Shader Callback
		0); //! No custom user data
	cb->drop();

	auto includes = m_rrManager->getRadeonRaysGLSLIncludes();
	m_raygenProgram = createComputeShader(
		raygenShaderExtensions+
		//"irr/builtin/glsl/ext/RadeonRays/"
		includes->getBuiltinInclude("ray.glsl")+
		raygenShader
	);
	m_compostProgram = createComputeShader(
		compostShaderExtensions+
		//"irr/builtin/glsl/ext/RadeonRays/"
		includes->getBuiltinInclude("ray.glsl") +
		compostShader
	);
}

Renderer::~Renderer()
{
	if (m_raygenProgram)
		COpenGLExtensionHandler::extGlDeleteProgram(m_raygenProgram);
	if (m_compostProgram)
		COpenGLExtensionHandler::extGlDeleteProgram(m_compostProgram);
}


void Renderer::init(const SAssetBundle& meshes, uint32_t rayBufferSize)
{
	deinit();

	core::vector<SLight> lights;
	core::vector<uint32_t> lightCDF;
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
			if (instance.emitter.type != ext::MitsubaLoader::CElementEmitter::Type::INVALID)
			{
				assert(instance.emitter.type==ext::MitsubaLoader::CElementEmitter::Type::AREA);

				uint32_t totalTriangleCount = 0u;
				auto cpumesh = static_cast<ICPUMesh*>(cpuit->get());
				asset::IMeshManipulator::getPolyCount(totalTriangleCount, cpumesh);

				SLight light;
				light.setFactor(instance.emitter.area.radiance);
				light.analytical.transform = instance.tform;
				auto tmp =	core::transpose(core::matrix4SIMD(instance.tform.getSub3x3TransposeCofactors()));
				light.analytical.transformCofactors = tmp.extractSub3x4();

				bool bail = false;
				switch (shapeType)
				{
					case ext::MitsubaLoader::CElementShape::Type::CUBE:
						light.type = SLight::ET_CUBE;
						break;
					case ext::MitsubaLoader::CElementShape::Type::SPHERE:
						light.type = SLight::ET_ELLIPSOID;
						break;
					case ext::MitsubaLoader::CElementShape::Type::CYLINDER:
						light.type = SLight::ET_CYLINDER;
						break;
					case ext::MitsubaLoader::CElementShape::Type::RECTANGLE:
						light.type = SLight::ET_RECTANGLE;
						break;
					case ext::MitsubaLoader::CElementShape::Type::DISK:
						light.type = SLight::ET_DISK;
						break;
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

				auto addLight = [&](float approxArea) -> void
				{
					float weight = light.computeFlux(approxArea) * instance.emitter.area.samplingWeight;
					if (weight <= FLT_MIN)
						return;

					lightPDF.push_back(weight);
					lights.push_back(light);
				};
				auto areaFromTriangulationAndMakeMeshLight = [&]() -> float
				{
					double totalSurfaceArea = 0.0;

					uint32_t runningTriangleCount = 0u;
					for (auto i=0u; i<meshBufferCount; i++)
					{
						auto cpumb = cpumesh->getMeshBuffer(i);

						uint32_t triangleCount = 0u;
						asset::IMeshManipulator::getPolyCount(triangleCount,cpumb);
						for (auto triID=0u; triID<triangleCount; triID++)
						{
							auto triangle = asset::IMeshManipulator::getTriangleIndices(cpumb,triID);

							core::vectorSIMDf v[3];
							for (auto k=0u; k<3u; k++)
								v[k] = cpumb->getPosition(triangle[k]);
						
							float triangleArea = 0.5f*light.computeAreaUnderTransform(core::cross(v[1]-v[0],v[2]-v[0]));

							if (light.type==SLight::ET_TRIANGLE)
							{
								for (auto k=0u; k<3u; k++)
									instance.tform.transformVect(v[k]);
								std::copy(v,v+3u,light.triangle.vertices);

								bool notLastTriangle = (++runningTriangleCount)!=totalTriangleCount;
								if (notLastTriangle) // so we don't double add
									addLight(triangleArea);
								else
									totalSurfaceArea = triangleArea;
							}
							else
								totalSurfaceArea += triangleArea;
						}
					}
					return totalSurfaceArea;
				};

				addLight(areaFromTriangulationAndMakeMeshLight());
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
		for (auto emitter : globalMeta->emitters)
		{
			SLight light;

			bool bail = false;
			float weight = 1.f;
			switch (emitter.type)
			{
				case ext::MitsubaLoader::CElementEmitter::Type::CONSTANT:
					constantClearColor += emitter.constant.radiance;
					light.type = SLight::ET_CONSTANT;
					light.setFactor(emitter.constant.radiance);
					weight = emitter.constant.samplingWeight;
					break;
				case ext::MitsubaLoader::CElementEmitter::Type::INVALID:
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
			
			weight *= light.computeFlux(NAN);
			if (weight <= FLT_MIN)
				continue;

			lightPDF.push_back(weight);
			lights.push_back(light);
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
		assert(weightSum >FLT_MIN);
		double weightSumRcp = double(~0u)/weightSum+double(FLT_MIN);

		double partialSum = 0.0;
		for (auto i=0u; i<lightCDF.size(); i++)
		{
			float pdf = lightPDF[i];
			partialSum += pdf;
			lightCDF[i] = static_cast<uint32_t>(partialSum*weightSumRcp);
			lights[i].setFactor(core::vectorSIMDf(lights[i].strengthFactor)*(weightSum/pdf));
		}

		m_lightCDFBuffer = core::smart_refctd_ptr<IGPUBuffer>(m_driver->createFilledDeviceLocalGPUBufferOnDedMem(lightCDF.size()*sizeof(uint32_t),lightCDF.data()),core::dont_grab);
		m_lightBuffer = core::smart_refctd_ptr<IGPUBuffer>(m_driver->createFilledDeviceLocalGPUBufferOnDedMem(lights.size()*sizeof(SLight),lights.data()),core::dont_grab);
	}

	m_depth = m_driver->createGPUTexture(ITexture::ETT_2D, &renderSize.Width, 1, EF_D32_SFLOAT);
	m_albedo = m_driver->createGPUTexture(ITexture::ETT_2D, &renderSize.Width, 1, EF_R8G8B8_SRGB);
	m_normals = m_driver->createGPUTexture(ITexture::ETT_2D, &renderSize.Width, 1, EF_R16G16_SNORM);

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

	// start deleting objects
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

	m_depth = m_albedo = m_normals = m_accumulation = m_tonemapOutput = nullptr;

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
	m_totalSamplesComputed = 0u;
	m_rayCount = 0u;
	m_framesDone = 0u;
	m_workGroupCount[0] = m_workGroupCount[1] = 0u;

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
	m_lightCDFBuffer = m_lightBuffer = nullptr;
}


void Renderer::render()
{
	m_driver->setRenderTarget(m_gbuffer);
	{ // clear
		m_driver->clearZBuffer();
		float zero[4] = { 0.f,0.f,0.f,0.f };
		m_driver->clearColorBuffer(EFAP_COLOR_ATTACHMENT0, zero);
		m_driver->clearColorBuffer(EFAP_COLOR_ATTACHMENT1, zero);
	}

	auto camera = m_smgr->getActiveCamera();
	auto prevViewProj = camera->getConcatenatedMatrix();

	//! This animates (moves) the camera and sets the transforms
	//! Also draws the meshbuffer
	m_smgr->drawAll();

	auto currentViewProj = camera->getConcatenatedMatrix();
	if (!core::equals(prevViewProj,currentViewProj,core::ROUNDING_ERROR<core::matrix4SIMD>()*100.0))
	{
		m_totalSamplesComputed = 0u;
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

			COpenGLExtensionHandler::pGlProgramUniform1uiv(m_raygenProgram, 5, 1, &m_totalSamplesComputed);
			m_totalSamplesComputed += m_rayCount;

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

		const COpenGLDriver::SAuxContext* foundConst = static_cast<COpenGLDriver*>(m_driver)->getThreadContext();
		COpenGLDriver::SAuxContext* found = const_cast<COpenGLDriver::SAuxContext*>(foundConst);
		
		COpenGLExtensionHandler::extGlBindImageTexture(0u,static_cast<COpenGLFilterableTexture*>(m_accumulation.get())->getOpenGLName(),0,false,0,GL_READ_WRITE,GL_RGBA32F);

		COpenGLExtensionHandler::extGlUseProgram(m_compostProgram);

		const COpenGLBuffer* buffers[] = { static_cast<const COpenGLBuffer*>(m_rayBuffer.get()),static_cast<const COpenGLBuffer*>(m_intersectionBuffer.get()) };
		ptrdiff_t offsets[] = { 0,0 };
		ptrdiff_t sizes[] = { m_rayBuffer->getSize(),m_intersectionBuffer->getSize() };
		found->setActiveSSBO(0, 2, buffers, offsets, sizes);

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