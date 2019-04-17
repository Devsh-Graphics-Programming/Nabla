// Copyright (C) 2002-2012 Nikolaus Gebhardt / Thomas Alten
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "IrrCompileConfig.h"
#include "CSoftwareDriver2.h"

#ifdef _IRR_COMPILE_WITH_BURNINGSVIDEO_

#include "SoftwareDriver2_helper.h"
#include "CSoftwareTexture2.h"
#include "CSoftware2MaterialRenderer.h"

#include "S4DVertex.h"
#include "CBlit.h"


#define MAT_TEXTURE(tex) ( (video::CSoftwareTexture2*) Material.org.getTexture ( tex ) )


namespace irr
{
namespace video
{

namespace glsl
{

typedef sVec4 vec4;
typedef sVec3 vec3;
typedef sVec2 vec2;

#define in
#define uniform
#define attribute
#define varying

#ifdef _MSC_VER
#pragma warning(disable:4244)
#endif

struct mat4{
   float m[4][4];

   vec4 operator* ( const vec4 &in ) const
   {
	   vec4 out;
	   return out;
   }

};

struct mat3{
   float m[3][3];

   vec3 operator* ( const vec3 &in ) const
   {
	   vec3 out;
	   return out;
   }
};

const int gl_MaxLights = 8;


inline float dot (float x, float y) { return x * y; }
inline float dot ( const vec2 &x, const vec2 &y) { return x.x * y.x + x.y * y.y; }
inline float dot ( const vec3 &x, const vec3 &y) { return x.x * y.x + x.y * y.y + x.z * y.z; }
inline float dot ( const vec4 &x, const vec4 &y) { return x.x * y.x + x.y * y.y + x.z * y.z + x.w * y.w; }

inline float reflect (float I, float N)				{ return I - 2.0 * dot (N, I) * N; }
inline vec2 reflect (const vec2 &I, const vec2 &N)	{ return I - N * 2.0 * dot (N, I); }
inline vec3 reflect (const vec3 &I, const vec3 &N)	{ return I - N * 2.0 * dot (N, I); }
inline vec4 reflect (const vec4 &I, const vec4 &N)	{ return I - N * 2.0 * dot (N, I); }


inline float refract (float I, float N, float eta){
    const float k = 1.0 - eta * eta * (1.0 - dot (N, I) * dot (N, I));
    if (k < 0.0)
        return 0.0;
    return eta * I - (eta * dot (N, I) + sqrt (k)) * N;
}

inline vec2 refract (const vec2 &I, const vec2 &N, float eta){
    const float k = 1.0 - eta * eta * (1.0 - dot (N, I) * dot (N, I));
    if (k < 0.0)
        return vec2 (0.0);
    return I * eta - N * (eta * dot (N, I) + sqrt (k));
}

inline vec3 refract (const vec3 &I, const vec3 &N, float eta) {
    const float k = 1.0 - eta * eta * (1.0 - dot (N, I) * dot (N, I));
    if (k < 0.0)
        return vec3 (0.0);
    return I * eta - N * (eta * dot (N, I) + sqrt (k));
}

inline vec4 refract (const vec4 &I, const vec4 &N, float eta) {
    const float k = 1.0 - eta * eta * (1.0 - dot (N, I) * dot (N, I));
    if (k < 0.0)
        return vec4 (0.0);
    return I * eta - N * (eta * dot (N, I) + sqrt (k));
}


inline float length ( const vec3 &v ) { return sqrtf ( v.x * v.x + v.y * v.y + v.z * v.z ); }
vec3 normalize ( const vec3 &v ) { 	float l = 1.f / length ( v ); return vec3 ( v.x * l, v.y * l, v.z * l ); }
float max ( float a, float b ) { return a > b ? a : b; }
float min ( float a, float b ) { return a < b ? a : b; }
vec4 clamp ( const vec4 &a, float low, float high ) { return vec4 ( min (max(a.x,low), high), min (max(a.y,low), high), min (max(a.z,low), high), min (max(a.w,low), high) ); }



typedef int sampler2D;
sampler2D texUnit0;

vec4 texture2D (sampler2D sampler, const vec2 &coord) { return vec4 (0.0); }

struct gl_LightSourceParameters {
	vec4 ambient;              // Acli
	vec4 diffuse;              // Dcli
	vec4 specular;             // Scli
	vec4 position;             // Ppli
	vec4 halfVector;           // Derived: Hi
	vec3 spotDirection;        // Sdli
	float spotExponent;        // Srli
	float spotCutoff;          // Crli
							// (range: [0.0,90.0], 180.0)
	float spotCosCutoff;       // Derived: cos(Crli)
							// (range: [1.0,0.0],-1.0)
	float constantAttenuation; // K0
	float linearAttenuation;   // K1
	float quadraticAttenuation;// K2
};

uniform gl_LightSourceParameters gl_LightSource[gl_MaxLights];

struct gl_LightModelParameters {
    vec4 ambient;
};
uniform gl_LightModelParameters gl_LightModel;

struct gl_LightModelProducts {
    vec4 sceneColor;
};

uniform gl_LightModelProducts gl_FrontLightModelProduct;
uniform gl_LightModelProducts gl_BackLightModelProduct;

struct gl_LightProducts {
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
};

uniform gl_LightProducts gl_FrontLightProduct[gl_MaxLights];
uniform gl_LightProducts gl_BackLightProduct[gl_MaxLights];

struct gl_MaterialParameters
{
	vec4 emission;    // Ecm
	vec4 ambient;     // Acm
	vec4 diffuse;     // Dcm
	vec4 specular;    // Scm
	float shininess;  // Srm
};
uniform gl_MaterialParameters gl_FrontMaterial;
uniform gl_MaterialParameters gl_BackMaterial;

// GLSL has some built-in attributes in a vertex shader:
attribute vec4 gl_Vertex;			// 4D vector representing the vertex position
attribute vec3 gl_Normal;			// 3D vector representing the vertex normal
attribute vec4 gl_Color;			// 4D vector representing the vertex color
attribute vec4 gl_MultiTexCoord0;	// 4D vector representing the texture coordinate of texture unit X
attribute vec4 gl_MultiTexCoord1;	// 4D vector representing the texture coordinate of texture unit X

uniform mat4 gl_ModelViewMatrix;			//4x4 Matrix representing the model-view matrix.
uniform mat4 gl_ModelViewProjectionMatrix;	//4x4 Matrix representing the model-view-projection matrix.
uniform mat3 gl_NormalMatrix;				//3x3 Matrix representing the inverse transpose model-view matrix. This matrix is used for normal transformation.


varying vec4 gl_FrontColor;				// 4D vector representing the primitives front color
varying vec4 gl_FrontSecondaryColor;	// 4D vector representing the primitives second front color
varying vec4 gl_BackColor;				// 4D vector representing the primitives back color
varying vec4 gl_TexCoord[4];			// 4D vector representing the Xth texture coordinate

// shader output
varying vec4 gl_Position;				// 4D vector representing the final processed vertex position. Only  available in vertex shader.
varying vec4 gl_FragColor;				// 4D vector representing the final color which is written in the frame buffer. Only available in fragment shader.
varying float gl_FragDepth;				// float representing the depth which is written in the depth buffer. Only available in fragment shader.

varying vec4 gl_SecondaryColor;
varying float gl_FogFragCoord;


vec4 ftransform(void)
{
	return gl_ModelViewProjectionMatrix * gl_Vertex;
}

vec3 fnormal(void)
{
    //Compute the normal
    vec3 normal = gl_NormalMatrix * gl_Normal;
    normal = normalize(normal);
    return normal;
}


struct program1
{
	vec4 Ambient;
	vec4 Diffuse;
	vec4 Specular;

	void pointLight(in int i, in vec3 normal, in vec3 eye, in vec3 ecPosition3)
	{
	   float nDotVP;       // normal . light direction
	   float nDotHV;       // normal . light half vector
	   float pf;           // power factor
	   float attenuation;  // computed attenuation factor
	   float d;            // distance from surface to light source
	   vec3  VP;           // direction from surface to light position
	   vec3  halfVector;   // direction of maximum highlights

	   // Compute vector from surface to light position
	   VP = vec3 (gl_LightSource[i].position) - ecPosition3;

	   // Compute distance between surface and light position
	   d = length(VP);

	   // Normalize the vector from surface to light position
	   VP = normalize(VP);

	   // Compute attenuation
	   attenuation = 1.0 / (gl_LightSource[i].constantAttenuation +
		   gl_LightSource[i].linearAttenuation * d +
		   gl_LightSource[i].quadraticAttenuation * d * d);

	   halfVector = normalize(VP + eye);

	   nDotVP = max(0.0, dot(normal, VP));
	   nDotHV = max(0.0, dot(normal, halfVector));

	   if (nDotVP == 0.0)
	   {
		   pf = 0.0;
	   }
	   else
	   {
		   pf = pow(nDotHV, gl_FrontMaterial.shininess);

	   }
	   Ambient  += gl_LightSource[i].ambient * attenuation;
	   Diffuse  += gl_LightSource[i].diffuse * nDotVP * attenuation;
	   Specular += gl_LightSource[i].specular * pf * attenuation;
	}

	vec3 fnormal(void)
	{
		//Compute the normal
		vec3 normal = gl_NormalMatrix * gl_Normal;
		normal = normalize(normal);
		return normal;
	}

	void ftexgen(in vec3 normal, in vec4 ecPosition)
	{

		gl_TexCoord[0] = gl_MultiTexCoord0;
	}

	void flight(in vec3 normal, in vec4 ecPosition, float alphaFade)
	{
		vec4 color;
		vec3 ecPosition3;
		vec3 eye;

		ecPosition3 = (vec3 (ecPosition)) / ecPosition.w;
		eye = vec3 (0.0, 0.0, 1.0);

		// Clear the light intensity accumulators
		Ambient  = vec4 (0.0);
		Diffuse  = vec4 (0.0);
		Specular = vec4 (0.0);

		pointLight(0, normal, eye, ecPosition3);

		pointLight(1, normal, eye, ecPosition3);

		color = gl_FrontLightModelProduct.sceneColor +
		  Ambient  * gl_FrontMaterial.ambient +
		  Diffuse  * gl_FrontMaterial.diffuse;
		gl_FrontSecondaryColor = Specular * gl_FrontMaterial.specular;
		color = clamp( color, 0.0, 1.0 );
		gl_FrontColor = color;

		gl_FrontColor.a *= alphaFade;
	}


	void vertexshader_main (void)
	{
		vec3  transformedNormal;
		float alphaFade = 1.0;

		// Eye-coordinate position of vertex, needed in various calculations
		vec4 ecPosition = gl_ModelViewMatrix * gl_Vertex;

		// Do fixed functionality vertex transform
		gl_Position = ftransform();
		transformedNormal = fnormal();
		flight(transformedNormal, ecPosition, alphaFade);
		ftexgen(transformedNormal, ecPosition);
	}

	void fragmentshader_main (void)
	{
		vec4 color;

		color = gl_Color;

		color *= texture2D(texUnit0, vec2(gl_TexCoord[0].x, gl_TexCoord[0].y) );

		color += gl_SecondaryColor;
		color = clamp(color, 0.0, 1.0);

		gl_FragColor = color;
	}
};

}

//! constructor
CBurningVideoDriver::CBurningVideoDriver(IrrlichtDevice* dev, const irr::SIrrlichtCreationParameters& params, io::IFileSystem* io, video::IImagePresenter* presenter)
: CNullDriver(dev, io, params.WindowSize), BackBuffer(0), Presenter(presenter),
	WindowId(0), SceneSourceRect(0),
	RenderTargetTexture(0), RenderTargetSurface(0), CurrentShader(0),
	 DepthBuffer(0), StencilBuffer ( 0 ),
	 CurrentOut ( 12 * 2, 128 ), Temp ( 12 * 2, 128 )
{
	#ifdef _IRR_DEBUG
	setDebugName("CBurningVideoDriver");
	#endif

	// create backbuffer
	BackBuffer = new CImage(BURNINGSHADER_COLOR_FORMAT, params.WindowSize);
	if (BackBuffer)
	{
		BackBuffer->fill(SColor(0));

		// create z buffer
		if ( params.ZBufferBits )
			DepthBuffer = video::createDepthBuffer(BackBuffer->getDimension());

		// create stencil buffer
		if ( params.Stencilbuffer )
			StencilBuffer = video::createStencilBuffer(BackBuffer->getDimension());
	}

	// create triangle renderers

	irr::memset32 ( BurningShader, 0, sizeof ( BurningShader ) );
	BurningShader[ETR_GOURAUD] = createTriangleRendererGouraud2(this);
	BurningShader[ETR_TEXTURE_GOURAUD] = createTriangleRendererTextureGouraud2(this);

	BurningShader[ETR_TEXTURE_GOURAUD_NOZ] = createTRTextureGouraudNoZ2(this);
	BurningShader[ETR_TEXTURE_GOURAUD_ADD] = createTRTextureGouraudAdd2(this);
	BurningShader[ETR_TEXTURE_GOURAUD_ADD_NO_Z] = createTRTextureGouraudAddNoZ2(this);
	BurningShader[ETR_TEXTURE_GOURAUD_VERTEX_ALPHA] = createTriangleRendererTextureVertexAlpha2 ( this );

	BurningShader[ETR_TEXTURE_GOURAUD_ALPHA] = createTRTextureGouraudAlpha(this );
	BurningShader[ETR_TEXTURE_GOURAUD_ALPHA_NOZ] = createTRTextureGouraudAlphaNoZ( this );

	BurningShader[ETR_REFERENCE] = createTriangleRendererReference ( this );



	// add the same renderer for all solid types
	CSoftware2MaterialRenderer_SOLID* smr = new CSoftware2MaterialRenderer_SOLID( this);
	CSoftware2MaterialRenderer_TRANSPARENT_ADD_COLOR* tmr = new CSoftware2MaterialRenderer_TRANSPARENT_ADD_COLOR( this);
	CSoftware2MaterialRenderer_UNSUPPORTED * umr = new CSoftware2MaterialRenderer_UNSUPPORTED ( this );

	//!TODO: addMaterialRenderer depends on pushing order....
	addMaterialRenderer ( smr ); // EMT_SOLID
	addMaterialRenderer ( tmr ); // EMT_TRANSPARENT_ADD_COLOR,
	addMaterialRenderer ( tmr ); // EMT_TRANSPARENT_ALPHA_CHANNEL,
	addMaterialRenderer ( tmr ); // EMT_TRANSPARENT_VERTEX_ALPHA,

	smr->drop ();
	tmr->drop ();
	umr->drop ();

	// select render target
	setRenderTarget(BackBuffer);

	//reset Lightspace
	LightSpace.reset ();

	// select the right renderer
	setCurrentShader();
}


//! destructor
CBurningVideoDriver::~CBurningVideoDriver()
{
	// delete Backbuffer
	if (BackBuffer)
		BackBuffer->drop();

	// delete triangle renderers

	for (int32_t i=0; i<ETR2_COUNT; ++i)
	{
		if (BurningShader[i])
			BurningShader[i]->drop();
	}

	// delete Additional buffer
	if (StencilBuffer)
		StencilBuffer->drop();

	if (DepthBuffer)
		DepthBuffer->drop();

	if (RenderTargetTexture)
		RenderTargetTexture->drop();

	if (RenderTargetSurface)
		RenderTargetSurface->drop();
}


/*!
	selects the right triangle renderer based on the render states.
*/
void CBurningVideoDriver::setCurrentShader()
{
	IVirtualTexture *texture0 = Material.org.getTexture(0);

	bool zMaterialTest =	Material.org.ZBuffer != ECFN_NEVER &&
							Material.org.ZWriteEnable &&
							( AllowZWriteOnTransparent || !Material.org.isTransparent() );

	EBurningFFShader shader = zMaterialTest ? ETR_TEXTURE_GOURAUD : ETR_TEXTURE_GOURAUD_NOZ;

	LightSpace.Flags &= ~VERTEXTRANSFORM;

	switch ( Material.org.MaterialType )
	{
		case EMT_TRANSPARENT_ALPHA_CHANNEL:
			if ( texture0 && texture0->hasAlpha () )
			{
				shader = zMaterialTest ? ETR_TEXTURE_GOURAUD_ALPHA : ETR_TEXTURE_GOURAUD_ALPHA_NOZ;
				break;
			}
			// fall through

		case EMT_TRANSPARENT_ADD_COLOR:
			shader = zMaterialTest ? ETR_TEXTURE_GOURAUD_ADD : ETR_TEXTURE_GOURAUD_ADD_NO_Z;
			break;

		case EMT_TRANSPARENT_VERTEX_ALPHA:
			shader = ETR_TEXTURE_GOURAUD_VERTEX_ALPHA;
			break;

		default:
			break;

	}

	if ( !texture0 )
        shader = ETR_GOURAUD;
    else
	{
	    switch (texture0->getVirtualTextureType())
	    {
            case IVirtualTexture::EVTT_OPAQUE_FILTERABLE:
	        case IVirtualTexture::EVTT_VIEW:
                break;
            default:
                shader = ETR_GOURAUD;
                break;
	    }
	}

	//shader = ETR_REFERENCE;

	// switchToTriangleRenderer
	CurrentShader = BurningShader[shader];
	if ( CurrentShader )
	{
		CurrentShader->setZCompareFunc ( Material.org.ZBuffer );
		CurrentShader->setRenderTarget(RenderTargetSurface, ViewPort);
		CurrentShader->setMaterial ( Material );

		switch ( shader )
		{
			case ETR_TEXTURE_GOURAUD_ALPHA:
			case ETR_TEXTURE_GOURAUD_ALPHA_NOZ:
				CurrentShader->setParam ( 0, Material.org.MaterialTypeParam );
				break;
			default:
			break;
		}
	}

}



//! clears the zbuffer
bool CBurningVideoDriver::beginScene(bool backBuffer, bool zBuffer,
		SColor color, const SExposedVideoData& videoData,
		core::rect<int32_t>* sourceRect)
{
	CNullDriver::beginScene(backBuffer, zBuffer, color, videoData, sourceRect);
	WindowId = videoData.OpenGLWin32.HWnd;
	SceneSourceRect = sourceRect;

	if (backBuffer && BackBuffer)
		BackBuffer->fill(color);

	if (zBuffer && DepthBuffer)
		DepthBuffer->clear();

	return true;
}


//! presents the rendered scene on the screen, returns false if failed
bool CBurningVideoDriver::endScene()
{
	CNullDriver::endScene();

	return Presenter->present(BackBuffer, WindowId, SceneSourceRect);
}


//! sets a render target
bool CBurningVideoDriver::setRenderTarget(video::ITexture* texture, bool clearBackBuffer,
								 bool clearZBuffer, SColor color)
{
	if (texture && texture->getDriverType() != EDT_BURNINGSVIDEO)
	{
		os::Printer::log("Fatal Error: Tried to set a texture not owned by this driver.", ELL_ERROR);
		return false;
	}

	if (RenderTargetTexture)
		RenderTargetTexture->drop();

	RenderTargetTexture = texture;

	if (RenderTargetTexture)
	{
		RenderTargetTexture->grab();
		setRenderTarget(((CSoftwareTexture2*)RenderTargetTexture)->getTexture());
	}
	else
	{
		setRenderTarget(BackBuffer);
	}

	if (RenderTargetSurface && (clearBackBuffer || clearZBuffer))
	{
		if (clearZBuffer)
			DepthBuffer->clear();

		if (clearBackBuffer)
			RenderTargetSurface->fill( color );
	}

	return true;
}


//! sets a render target
void CBurningVideoDriver::setRenderTarget(video::CImage* image)
{
	if (RenderTargetSurface)
		RenderTargetSurface->drop();

	RenderTargetSurface = image;
	RenderTargetSize.Width = 0;
	RenderTargetSize.Height = 0;

	if (RenderTargetSurface)
	{
		RenderTargetSurface->grab();
		RenderTargetSize = RenderTargetSurface->getDimension();
	}

	setViewPort(core::rect<int32_t>(0,0,RenderTargetSize.Width,RenderTargetSize.Height));

	if (DepthBuffer)
		DepthBuffer->setSize(RenderTargetSize);

	if (StencilBuffer)
		StencilBuffer->setSize(RenderTargetSize);
}



//! sets a viewport
void CBurningVideoDriver::setViewPort(const core::rect<int32_t>& area)
{
	ViewPort = area;

	core::rect<int32_t> rendert(0,0,RenderTargetSize.Width,RenderTargetSize.Height);
	ViewPort.clipAgainst(rendert);

	auto buildNDCToDCMatrix = []( const core::rect<int32_t>& viewport)
	{
	    // wtf is with the 0.75 ?
		const float scaleX = (viewport.getWidth() - 0.75f ) * 0.5f;
		const float scaleY = -(viewport.getHeight() - 0.75f ) * 0.5f;

		const float dx = -0.5f + ( (viewport.UpperLeftCorner.X + viewport.LowerRightCorner.X ) * 0.5f );
		const float dy = -0.5f + ( (viewport.UpperLeftCorner.Y + viewport.LowerRightCorner.Y ) * 0.5f );

		core::matrix4SIMD retval;
		retval.setScale(core::vectorSIMDf(scaleX, scaleY, 1.f));
		retval.setTranslation(core::vectorSIMDf(dx, dy, 0.f));
		return retval;
	};

	ClipscaleTransformation = buildNDCToDCMatrix(ViewPort);

	if (CurrentShader)
		CurrentShader->setRenderTarget(RenderTargetSurface, ViewPort);
}

/*
	generic plane clipping in homogenous coordinates
	special case ndc frustum <-w,w>,<-w,w>,<-w,w>
	can be rewritten with compares e.q near plane, a.z < -a.w and b.z < -b.w
*/

const sVec4 CBurningVideoDriver::NDCPlane[6] =
{
	sVec4(  0.f,  0.f, -1.f, -1.f ),	// near
	sVec4(  0.f,  0.f,  1.f, -1.f ),	// far
	sVec4(  1.f,  0.f,  0.f, -1.f ),	// left
	sVec4( -1.f,  0.f,  0.f, -1.f ),	// right
	sVec4(  0.f,  1.f,  0.f, -1.f ),	// bottom
	sVec4(  0.f, -1.f,  0.f, -1.f )		// top
};



/*
	test a vertex if it's inside the standard frustum

	this is the generic one..

	float dotPlane;
	for ( uint32_t i = 0; i!= 6; ++i )
	{
		dotPlane = v->Pos.dotProduct ( NDCPlane[i] );
		core::setbit_cond( flag, dotPlane <= 0.f, 1 << i );
	}

	// this is the base for ndc frustum <-w,w>,<-w,w>,<-w,w>
	core::setbit_cond( flag, ( v->Pos.z - v->Pos.w ) <= 0.f, 1 );
	core::setbit_cond( flag, (-v->Pos.z - v->Pos.w ) <= 0.f, 2 );
	core::setbit_cond( flag, ( v->Pos.x - v->Pos.w ) <= 0.f, 4 );
	core::setbit_cond( flag, (-v->Pos.x - v->Pos.w ) <= 0.f, 8 );
	core::setbit_cond( flag, ( v->Pos.y - v->Pos.w ) <= 0.f, 16 );
	core::setbit_cond( flag, (-v->Pos.y - v->Pos.w ) <= 0.f, 32 );

*/
#ifdef __IRR_FAST_MATH

REALINLINE uint32_t CBurningVideoDriver::clipToFrustumTest ( const s4DVertex * v  ) const
{
	float test[6];
	uint32_t flag;
	const float w = - v->Pos.w;

	// a conditional move is needed....FCOMI ( but we don't have it )
	// so let the fpu calculate and write it back.
	// cpu makes the compare, interleaving

	test[0] =  v->Pos.z + w;
	test[1] = -v->Pos.z + w;
	test[2] =  v->Pos.x + w;
	test[3] = -v->Pos.x + w;
	test[4] =  v->Pos.y + w;
	test[5] = -v->Pos.y + w;

	flag  = (IR ( test[0] )              ) >> 31;
	flag |= (IR ( test[1] ) & 0x80000000 ) >> 30;
	flag |= (IR ( test[2] ) & 0x80000000 ) >> 29;
	flag |= (IR ( test[3] ) & 0x80000000 ) >> 28;
	flag |= (IR ( test[4] ) & 0x80000000 ) >> 27;
	flag |= (IR ( test[5] ) & 0x80000000 ) >> 26;

/*
	flag  = F32_LOWER_EQUAL_0 ( test[0] );
	flag |= F32_LOWER_EQUAL_0 ( test[1] ) << 1;
	flag |= F32_LOWER_EQUAL_0 ( test[2] ) << 2;
	flag |= F32_LOWER_EQUAL_0 ( test[3] ) << 3;
	flag |= F32_LOWER_EQUAL_0 ( test[4] ) << 4;
	flag |= F32_LOWER_EQUAL_0 ( test[5] ) << 5;
*/
	return flag;
}

#else


REALINLINE uint32_t CBurningVideoDriver::clipToFrustumTest ( const s4DVertex * v  ) const
{
	uint32_t flag = 0;

	if ( v->Pos.z <= v->Pos.w ) flag |= 1;
	if (-v->Pos.z <= v->Pos.w ) flag |= 2;

	if ( v->Pos.x <= v->Pos.w ) flag |= 4;
	if (-v->Pos.x <= v->Pos.w ) flag |= 8;

	if ( v->Pos.y <= v->Pos.w ) flag |= 16;
	if (-v->Pos.y <= v->Pos.w ) flag |= 32;

/*
	for ( uint32_t i = 0; i!= 6; ++i )
	{
		core::setbit_cond( flag, v->Pos.dotProduct ( NDCPlane[i] ) <= 0.f, 1 << i );
	}
*/
	return flag;
}

#endif // __IRR_FAST_MATH

uint32_t CBurningVideoDriver::clipToHyperPlane ( s4DVertex * dest, const s4DVertex * source, uint32_t inCount, const sVec4 &plane )
{
	uint32_t outCount = 0;
	s4DVertex * out = dest;

	const s4DVertex * a;
	const s4DVertex * b = source;

	float bDotPlane;

	bDotPlane = b->Pos.dotProduct ( plane );

	for( uint32_t i = 1; i < inCount + 1; ++i)
	{
		const int32_t condition = i - inCount;
		const int32_t index = (( ( condition >> 31 ) & ( i ^ condition ) ) ^ condition ) << 1;

		a = &source[ index ];

		// current point inside
		if ( a->Pos.dotProduct ( plane ) <= 0.f )
		{
			// last point outside
			if ( F32_GREATER_0 ( bDotPlane ) )
			{
				// intersect line segment with plane
				out->interpolate ( *b, *a, bDotPlane / (b->Pos - a->Pos).dotProduct ( plane ) );
				out += 2;
				outCount += 1;
			}

			// copy current to out
			//*out = *a;
			irr::memcpy32_small ( out, a, SIZEOF_SVERTEX * 2 );
			b = out;

			out += 2;
			outCount += 1;
		}
		else
		{
			// current point outside

			if ( F32_LOWER_EQUAL_0 (  bDotPlane ) )
			{
				// previous was inside
				// intersect line segment with plane
				out->interpolate ( *b, *a, bDotPlane / (b->Pos - a->Pos).dotProduct ( plane ) );
				out += 2;
				outCount += 1;
			}
			// pointer
			b = a;
		}

		bDotPlane = b->Pos.dotProduct ( plane );

	}

	return outCount;
}


uint32_t CBurningVideoDriver::clipToFrustum ( s4DVertex *v0, s4DVertex * v1, const uint32_t vIn )
{
	uint32_t vOut = vIn;

	vOut = clipToHyperPlane ( v1, v0, vOut, NDCPlane[0] ); if ( vOut < vIn ) return vOut;
	vOut = clipToHyperPlane ( v0, v1, vOut, NDCPlane[1] ); if ( vOut < vIn ) return vOut;
	vOut = clipToHyperPlane ( v1, v0, vOut, NDCPlane[2] ); if ( vOut < vIn ) return vOut;
	vOut = clipToHyperPlane ( v0, v1, vOut, NDCPlane[3] ); if ( vOut < vIn ) return vOut;
	vOut = clipToHyperPlane ( v1, v0, vOut, NDCPlane[4] ); if ( vOut < vIn ) return vOut;
	vOut = clipToHyperPlane ( v0, v1, vOut, NDCPlane[5] );
	return vOut;
}

/*!
 Part I:
	apply Clip Scale matrix
	From Normalized Device Coordiante ( NDC ) Space to Device Coordinate Space ( DC )

 Part II:
	Project homogeneous vector
	homogeneous to non-homogenous coordinates ( dividebyW )

	Incoming: ( xw, yw, zw, w, u, v, 1, R, G, B, A )
	Outgoing: ( xw/w, yw/w, zw/w, w/w, u/w, v/w, 1/w, R/w, G/w, B/w, A/w )


	replace w/w by 1/w
*/
inline void CBurningVideoDriver::ndc_2_dc_and_project ( s4DVertex *dest,s4DVertex *source, uint32_t vIn ) const
{
	uint32_t g;

	for ( g = 0; g != vIn; g += 2 )
	{
		if ( (dest[g].flag & VERTEX4D_PROJECTED ) == VERTEX4D_PROJECTED )
			continue;

		dest[g].flag = source[g].flag | VERTEX4D_PROJECTED;

		const float w = source[g].Pos.w;
		const float iw = core::reciprocal ( w );

		// to device coordinates
		dest[g].Pos.x = iw * ( source[g].Pos.x * ClipscaleTransformation(0,0) + w * ClipscaleTransformation(0,3) );
		dest[g].Pos.y = iw * ( source[g].Pos.y * ClipscaleTransformation(1,1) + w * ClipscaleTransformation(1,3) );

#ifndef SOFTWARE_DRIVER_2_USE_WBUFFER
		dest[g].Pos.z = iw * source[g].Pos.z;
#endif

	#ifdef SOFTWARE_DRIVER_2_USE_VERTEX_COLOR
		#ifdef SOFTWARE_DRIVER_2_PERSPECTIVE_CORRECT
			dest[g].Color[0] = source[g].Color[0] * iw;
		#else
			dest[g].Color[0] = source[g].Color[0];
		#endif

	#endif
		dest[g].LightTangent[0] = source[g].LightTangent[0] * iw;
		dest[g].Pos.w = iw;
	}
}


inline void CBurningVideoDriver::ndc_2_dc_and_project2 ( const s4DVertex **v, const uint32_t size ) const
{
	uint32_t g;

	for ( g = 0; g != size; g += 1 )
	{
		s4DVertex * a = (s4DVertex*) v[g];

		if ( (a[1].flag & VERTEX4D_PROJECTED ) == VERTEX4D_PROJECTED )
			continue;

		a[1].flag = a->flag | VERTEX4D_PROJECTED;

		// project homogenous vertex, store 1/w
		const float w = a->Pos.w;
		const float iw = core::reciprocal ( w );

		// to device coordinates
		a[1].Pos.x = iw * ( a->Pos.x * ClipscaleTransformation(0,0) + w * ClipscaleTransformation(0,3) );
		a[1].Pos.y = iw * ( a->Pos.y * ClipscaleTransformation(1,1)+ w * ClipscaleTransformation(1,3) );

#ifndef SOFTWARE_DRIVER_2_USE_WBUFFER
		a[1].Pos.z = a->Pos.z * iw;
#endif

	#ifdef SOFTWARE_DRIVER_2_USE_VERTEX_COLOR
		#ifdef SOFTWARE_DRIVER_2_PERSPECTIVE_CORRECT
			a[1].Color[0] = a->Color[0] * iw;
		#else
			a[1].Color[0] = a->Color[0];
		#endif
	#endif

		a[1].LightTangent[0] = a[0].LightTangent[0] * iw;
		a[1].Pos.w = iw;

	}

}


/*!
	crossproduct in projected 2D -> screen area triangle
*/
inline float CBurningVideoDriver::screenarea ( const s4DVertex *v ) const
{
	return	( ( v[3].Pos.x - v[1].Pos.x ) * ( v[5].Pos.y - v[1].Pos.y ) ) -
			( ( v[3].Pos.y - v[1].Pos.y ) * ( v[5].Pos.x - v[1].Pos.x ) );
}


/*!
*/
inline float CBurningVideoDriver::texelarea ( const s4DVertex *v, int tex ) const
{
	float z;

	z =		( (v[2].Tex[tex].x - v[0].Tex[tex].x ) * (v[4].Tex[tex].y - v[0].Tex[tex].y ) )
		 -	( (v[4].Tex[tex].x - v[0].Tex[tex].x ) * (v[2].Tex[tex].y - v[0].Tex[tex].y ) );

	return MAT_TEXTURE ( tex )->getLODFactor ( z );
}

/*!
	crossproduct in projected 2D
*/
inline float CBurningVideoDriver::screenarea2 ( const s4DVertex **v ) const
{
	return	( (( v[1] + 1 )->Pos.x - (v[0] + 1 )->Pos.x ) * ( (v[2] + 1 )->Pos.y - (v[0] + 1 )->Pos.y ) ) -
			( (( v[1] + 1 )->Pos.y - (v[0] + 1 )->Pos.y ) * ( (v[2] + 1 )->Pos.x - (v[0] + 1 )->Pos.x ) );
}

/*!
*/
inline float CBurningVideoDriver::texelarea2 ( const s4DVertex **v, int32_t tex ) const
{
	float z;
	z =		( (v[1]->Tex[tex].x - v[0]->Tex[tex].x ) * (v[2]->Tex[tex].y - v[0]->Tex[tex].y ) )
		 -	( (v[2]->Tex[tex].x - v[0]->Tex[tex].x ) * (v[1]->Tex[tex].y - v[0]->Tex[tex].y ) );

	return MAT_TEXTURE ( tex )->getLODFactor ( z );
}


/*!
*/
inline void CBurningVideoDriver::select_polygon_mipmap ( s4DVertex *v, uint32_t vIn, uint32_t tex, const core::dimension2du& texSize ) const
{
	float f[2];

	f[0] = (float) texSize.Width - 0.25f;
	f[1] = (float) texSize.Height - 0.25f;

#ifdef SOFTWARE_DRIVER_2_PERSPECTIVE_CORRECT
	for ( uint32_t g = 0; g != vIn; g += 2 )
	{
		(v + g + 1 )->Tex[tex].x	= (v + g + 0)->Tex[tex].x * ( v + g + 1 )->Pos.w * f[0];
		(v + g + 1 )->Tex[tex].y	= (v + g + 0)->Tex[tex].y * ( v + g + 1 )->Pos.w * f[1];
	}
#else
	for ( uint32_t g = 0; g != vIn; g += 2 )
	{
		(v + g + 1 )->Tex[tex].x	= (v + g + 0)->Tex[tex].x * f[0];
		(v + g + 1 )->Tex[tex].y	= (v + g + 0)->Tex[tex].y * f[1];
	}
#endif
}

inline void CBurningVideoDriver::select_polygon_mipmap2 ( s4DVertex **v, uint32_t tex, const core::dimension2du& texSize ) const
{
	float f[2];

	f[0] = (float) texSize.Width - 0.25f;
	f[1] = (float) texSize.Height - 0.25f;

#ifdef SOFTWARE_DRIVER_2_PERSPECTIVE_CORRECT
	(v[0] + 1 )->Tex[tex].x	= v[0]->Tex[tex].x * ( v[0] + 1 )->Pos.w * f[0];
	(v[0] + 1 )->Tex[tex].y	= v[0]->Tex[tex].y * ( v[0] + 1 )->Pos.w * f[1];

	(v[1] + 1 )->Tex[tex].x	= v[1]->Tex[tex].x * ( v[1] + 1 )->Pos.w * f[0];
	(v[1] + 1 )->Tex[tex].y	= v[1]->Tex[tex].y * ( v[1] + 1 )->Pos.w * f[1];

	(v[2] + 1 )->Tex[tex].x	= v[2]->Tex[tex].x * ( v[2] + 1 )->Pos.w * f[0];
	(v[2] + 1 )->Tex[tex].y	= v[2]->Tex[tex].y * ( v[2] + 1 )->Pos.w * f[1];

#else
	(v[0] + 1 )->Tex[tex].x	= v[0]->Tex[tex].x * f[0];
	(v[0] + 1 )->Tex[tex].y	= v[0]->Tex[tex].y * f[1];

	(v[1] + 1 )->Tex[tex].x	= v[1]->Tex[tex].x * f[0];
	(v[1] + 1 )->Tex[tex].y	= v[1]->Tex[tex].y * f[1];

	(v[2] + 1 )->Tex[tex].x	= v[2]->Tex[tex].x * f[0];
	(v[2] + 1 )->Tex[tex].y	= v[2]->Tex[tex].y * f[1];
#endif
}

#ifndef NEW_MESHES
// Vertex Cache
const SVSize CBurningVideoDriver::vSize[] =
{
	{ VERTEX4D_FORMAT_TEXTURE_1 | VERTEX4D_FORMAT_COLOR_1, sizeof(S3DVertex), 1 },
	{ VERTEX4D_FORMAT_TEXTURE_2 | VERTEX4D_FORMAT_COLOR_1, sizeof(S3DVertex2TCoords),2 },
	{ VERTEX4D_FORMAT_TEXTURE_2 | VERTEX4D_FORMAT_COLOR_1 | VERTEX4D_FORMAT_BUMP_DOT3, sizeof(S3DVertexTangents),2 },
	{ VERTEX4D_FORMAT_TEXTURE_2 | VERTEX4D_FORMAT_COLOR_1, sizeof(S3DVertex), 2 },	// reflection map
	{ 0, sizeof(float) * 3, 0 },	// core::vector3df*
};



/*!
	fill a cache line with transformed, light and clipp test triangles
*/
void CBurningVideoDriver::VertexCache_fill(const uint32_t sourceIndex, const uint32_t destIndex)
{
	uint8_t * source;
	s4DVertex *dest;

	source = (uint8_t*) VertexCache.vertices + ( sourceIndex * vSize[VertexCache.vType].Pitch );

	// it's a look ahead so we never hit it..
	// but give priority...
	//VertexCache.info[ destIndex ].hit = hitCount;

	// store info
	VertexCache.info[ destIndex ].index = sourceIndex;
	VertexCache.info[ destIndex ].hit = 0;

	// destination Vertex
	dest = (s4DVertex *) ( (uint8_t*) VertexCache.mem.data + ( destIndex << ( SIZEOF_SVERTEX_LOG2 + 1  ) ) );

	// transform Model * World * Camera * Projection * NDCSpace matrix
	const S3DVertex *base = ((S3DVertex*) source );
	Transformation [ ETS_CURRENT].transformVect ( &dest->Pos.x, base->Pos );

	//mhm ;-) maybe no goto
	if ( VertexCache.vType == 4 ) goto clipandproject;


#if defined (SOFTWARE_DRIVER_2_LIGHTING) || defined ( SOFTWARE_DRIVER_2_TEXTURE_TRANSFORM )

	// vertex normal in light space
	if ( /*Material.org.Lighting ||*/ (LightSpace.Flags & VERTEXTRANSFORM) )
	{
        getTransform(E4X3TS_WORLD).mul3x3with3x1( &LightSpace.normal.x, base->Normal );

        // vertex in light space
        if ( LightSpace.Flags & ( POINTLIGHT | FOG | SPECULAR | VERTEXTRANSFORM) )
            getTransform(E4X3TS_WORLD).transformVect ( &LightSpace.vertex.x, &base->Pos.X );

		if ( LightSpace.Flags & NORMALIZE )
			LightSpace.normal.normalize_xyz();

	}

#endif

#if defined ( SOFTWARE_DRIVER_2_USE_VERTEX_COLOR )
	dest->Color[0].setA8R8G8B8 ( base->Color.color );
#endif

	// Texture Transform
#if !defined ( SOFTWARE_DRIVER_2_TEXTURE_TRANSFORM )
	irr::memcpy32_small ( &dest->Tex[0],&base->TCoords,
					vSize[VertexCache.vType].TexSize << 3 //  * ( sizeof ( float ) * 2 )
				);
#else

	if ( 0 == (LightSpace.Flags & VERTEXTRANSFORM) )
	{
		irr::memcpy32_small ( &dest->Tex[0],&base->TCoords,
						vSize[VertexCache.vType].TexSize << 3 //  * ( sizeof ( float ) * 2 )
					);
	}
	else
	{
	/*
			Generate texture coordinates as linear functions so that:
				u = Ux*x + Uy*y + Uz*z + Uw
				v = Vx*x + Vy*y + Vz*z + Vw
			The matrix M for this case is:
				Ux  Vx  0  0
				Uy  Vy  0  0
				Uz  Vz  0  0
				Uw  Vw  0  0
	*/

		uint32_t t;
		sVec4 n;
		sVec2 srcT;

		for ( t = 0; t != vSize[VertexCache.vType].TexSize; ++t )
		{
			{
				irr::memcpy32_small ( &srcT,(&base->TCoords) + t,
					sizeof ( float ) * 2 );
			}

			switch ( Material.org.TextureLayer[t].SamplingParams.TextureWrapU )
			{
				case ETC_CLAMP_TO_EDGE:
				case ETC_CLAMP_TO_BORDER:
					dest->Tex[t].x = core::clamp ( (float) ( srcT.x ), 0.f, 1.f );
					break;
				case ETC_MIRROR:
					dest->Tex[t].x = srcT.x;
					if (core::fract(dest->Tex[t].x)>0.5f)
						dest->Tex[t].x=1.f-dest->Tex[t].x;
				break;
				case ETC_MIRROR_CLAMP_TO_EDGE:
				case ETC_MIRROR_CLAMP_TO_BORDER:
					dest->Tex[t].x = core::clamp ( (float) ( srcT.x ), 0.f, 1.f );
					if (core::fract(dest->Tex[t].x)>0.5f)
						dest->Tex[t].x=1.f-dest->Tex[t].x;
				break;
				case ETC_REPEAT:
				default:
					dest->Tex[t].x = srcT.x;
					break;
			}
			switch ( Material.org.TextureLayer[t].SamplingParams.TextureWrapV )
			{
				case ETC_CLAMP_TO_EDGE:
				case ETC_CLAMP_TO_BORDER:
					dest->Tex[t].y = core::clamp ( (float) ( srcT.y ), 0.f, 1.f );
					break;
				case ETC_MIRROR:
					dest->Tex[t].y = srcT.y;
					if (core::fract(dest->Tex[t].y)>0.5f)
						dest->Tex[t].y=1.f-dest->Tex[t].y;
				break;
				case ETC_MIRROR_CLAMP_TO_EDGE:
				case ETC_MIRROR_CLAMP_TO_BORDER:
					dest->Tex[t].y = core::clamp ( (float) ( srcT.y ), 0.f, 1.f );
					if (core::fract(dest->Tex[t].y)>0.5f)
						dest->Tex[t].y=1.f-dest->Tex[t].y;
				break;
				case ETC_REPEAT:
				default:
					dest->Tex[t].y = srcT.y;
					break;
			}
		}
	}


	if ( LightSpace.Light.size () && ( vSize[VertexCache.vType].Format & VERTEX4D_FORMAT_BUMP_DOT3 ) )
	{
		const S3DVertexTangents *tangent = ((S3DVertexTangents*) source );

		sVec4 vp;

		dest->LightTangent[0].x = 0.f;
		dest->LightTangent[0].y = 0.f;
		dest->LightTangent[0].z = 0.f;
		for ( uint32_t i = 0; i < 2 && i < LightSpace.Light.size (); ++i )
		{
			const SBurningShaderLight &light = LightSpace.Light[i];

			if ( !light.LightIsOn )
				continue;

			vp.x = light.pos.x - LightSpace.vertex.x;
			vp.y = light.pos.y - LightSpace.vertex.y;
			vp.z = light.pos.z - LightSpace.vertex.z;

	/*
			vp.x = light.pos_objectspace.x - base->Pos.X;
			vp.y = light.pos_objectspace.y - base->Pos.Y;
			vp.z = light.pos_objectspace.z - base->Pos.Z;
	*/

			vp.normalize_xyz();


			// transform by tangent matrix
			sVec3 l;
			l.x = (vp.x * tangent->Tangent.X + vp.y * tangent->Tangent.Y + vp.z * tangent->Tangent.Z );
			l.y = (vp.x * tangent->Binormal.X + vp.y * tangent->Binormal.Y + vp.z * tangent->Binormal.Z );
			l.z = (vp.x * tangent->Normal.X + vp.y * tangent->Normal.Y + vp.z * tangent->Normal.Z );


	/*
			float scale = 1.f / 128.f;
			scale /= dest->LightTangent[0].b;

			// emboss, shift coordinates
			dest->Tex[1].x = dest->Tex[0].x + l.r * scale;
			dest->Tex[1].y = dest->Tex[0].y + l.g * scale;
	*/
			dest->Tex[1].x = dest->Tex[0].x;
			dest->Tex[1].y = dest->Tex[0].y;

			// scale bias
			dest->LightTangent[0].x += l.x;
			dest->LightTangent[0].y += l.y;
			dest->LightTangent[0].z += l.z;
		}
		dest->LightTangent[0].setLength ( 0.5f );
		dest->LightTangent[0].x += 0.5f;
		dest->LightTangent[0].y += 0.5f;
		dest->LightTangent[0].z += 0.5f;
	}


#endif

clipandproject:
	dest[0].flag = dest[1].flag = vSize[VertexCache.vType].Format;

	// test vertex
	dest[0].flag |= clipToFrustumTest ( dest);

	// to DC Space, project homogenous vertex
	if ( (dest[0].flag & VERTEX4D_CLIPMASK ) == VERTEX4D_INSIDE )
	{
		ndc_2_dc_and_project2 ( (const s4DVertex**) &dest, 1 );
	}

	//return dest;
}

//

REALINLINE s4DVertex * CBurningVideoDriver::VertexCache_getVertex ( const uint32_t sourceIndex )
{
	for ( int32_t i = 0; i < VERTEXCACHE_ELEMENT; ++i )
	{
		if ( VertexCache.info[ i ].index == sourceIndex )
		{
			return (s4DVertex *) ( (uint8_t*) VertexCache.mem.data + ( i << ( SIZEOF_SVERTEX_LOG2 + 1  ) ) );
		}
	}
	return 0;
}


/*
	Cache based on linear walk indices
	fill blockwise on the next 16(Cache_Size) unique vertices in indexlist
	merge the next 16 vertices with the current
*/
REALINLINE void CBurningVideoDriver::VertexCache_get(const s4DVertex ** face)
{
	SCacheInfo info[VERTEXCACHE_ELEMENT];

	// next primitive must be complete in cache
	if (	VertexCache.indicesIndex - VertexCache.indicesRun < 3 &&
			VertexCache.indicesIndex < VertexCache.indexCount
		)
	{
		// rewind to start of primitive
		VertexCache.indicesIndex = VertexCache.indicesRun;

		irr::memset32 ( info, VERTEXCACHE_MISS, sizeof ( info ) );

		// get the next unique vertices cache line
		uint32_t fillIndex = 0;
		uint32_t dIndex;
		uint32_t i;

		while ( VertexCache.indicesIndex < VertexCache.indexCount &&
				fillIndex < VERTEXCACHE_ELEMENT
				)
		{
            uint32_t sourceIndex;
			switch ( VertexCache.iType )
			{
				case 1:
					sourceIndex =  ((uint16_t*)VertexCache.indices) [ VertexCache.indicesIndex ];
					break;
				case 2:
					sourceIndex =  ((uint32_t*)VertexCache.indices) [ VertexCache.indicesIndex ];
					break;
				case 4:
					sourceIndex = VertexCache.indicesIndex;
					break;
			}

			VertexCache.indicesIndex += 1;

			// if not exist, push back
			int32_t exist = 0;
			for ( dIndex = 0;  dIndex < fillIndex; ++dIndex )
			{
				if ( info[ dIndex ].index == sourceIndex )
				{
					exist = 1;
					break;
				}
			}

			if ( 0 == exist )
			{
				info[fillIndex++].index = sourceIndex;
			}
		}

		// clear marks
		for ( i = 0; i!= VERTEXCACHE_ELEMENT; ++i )
		{
			VertexCache.info[i].hit = 0;
		}

		// mark all existing
		for ( i = 0; i!= fillIndex; ++i )
		{
			for ( dIndex = 0;  dIndex < VERTEXCACHE_ELEMENT; ++dIndex )
			{
				if ( VertexCache.info[ dIndex ].index == info[i].index )
				{
					info[i].hit = dIndex;
					VertexCache.info[ dIndex ].hit = 1;
					break;
				}
			}
		}

		// fill new
		for ( i = 0; i!= fillIndex; ++i )
		{
			if ( info[i].hit != VERTEXCACHE_MISS )
				continue;

			for ( dIndex = 0;  dIndex < VERTEXCACHE_ELEMENT; ++dIndex )
			{
				if ( 0 == VertexCache.info[dIndex].hit )
				{
					VertexCache_fill ( info[i].index, dIndex );
					VertexCache.info[dIndex].hit += 1;
					info[i].hit = dIndex;
					break;
				}
			}
		}
	}

	const uint32_t i0 = core::if_c_a_else_0 ( VertexCache.pType != scene::EPT_TRIANGLE_FAN, VertexCache.indicesRun );

	switch ( VertexCache.iType )
	{
		case 1:
		{
			const uint16_t *p = (const uint16_t *) VertexCache.indices;
			face[0] = VertexCache_getVertex ( p[ i0    ] );
			face[1] = VertexCache_getVertex ( p[ VertexCache.indicesRun + 1] );
			face[2] = VertexCache_getVertex ( p[ VertexCache.indicesRun + 2] );
		}
		break;

		case 2:
		{
			const uint32_t *p = (const uint32_t *) VertexCache.indices;
			face[0] = VertexCache_getVertex ( p[ i0    ] );
			face[1] = VertexCache_getVertex ( p[ VertexCache.indicesRun + 1] );
			face[2] = VertexCache_getVertex ( p[ VertexCache.indicesRun + 2] );
		}
		break;

		case 4:
			face[0] = VertexCache_getVertex ( VertexCache.indicesRun + 0 );
			face[1] = VertexCache_getVertex ( VertexCache.indicesRun + 1 );
			face[2] = VertexCache_getVertex ( VertexCache.indicesRun + 2 );
		break;
		default:
			face[0] = face[1] = face[2] = VertexCache_getVertex(VertexCache.indicesRun + 0);
		break;
	}

	VertexCache.indicesRun += VertexCache.primitivePitch;
}

/*!
*/
REALINLINE void CBurningVideoDriver::VertexCache_getbypass ( s4DVertex ** face )
{
	const uint32_t i0 = core::if_c_a_else_0 ( VertexCache.pType != scene::EPT_TRIANGLE_FAN, VertexCache.indicesRun );

	if ( VertexCache.iType == 1 )
	{
		const uint16_t *p = (const uint16_t *) VertexCache.indices;
		VertexCache_fill ( p[ i0    ], 0 );
		VertexCache_fill ( p[ VertexCache.indicesRun + 1], 1 );
		VertexCache_fill ( p[ VertexCache.indicesRun + 2], 2 );
	}
	else
	{
		const uint32_t *p = (const uint32_t *) VertexCache.indices;
		VertexCache_fill ( p[ i0    ], 0 );
		VertexCache_fill ( p[ VertexCache.indicesRun + 1], 1 );
		VertexCache_fill ( p[ VertexCache.indicesRun + 2], 2 );
	}

	VertexCache.indicesRun += VertexCache.primitivePitch;

	face[0] = (s4DVertex *) ( (uint8_t*) VertexCache.mem.data + ( 0 << ( SIZEOF_SVERTEX_LOG2 + 1  ) ) );
	face[1] = (s4DVertex *) ( (uint8_t*) VertexCache.mem.data + ( 1 << ( SIZEOF_SVERTEX_LOG2 + 1  ) ) );
	face[2] = (s4DVertex *) ( (uint8_t*) VertexCache.mem.data + ( 2 << ( SIZEOF_SVERTEX_LOG2 + 1  ) ) );

}

/*!
*/
void CBurningVideoDriver::VertexCache_reset ( const void* vertices, uint32_t vertexCount,
											const void* indices, uint32_t primitiveCount,
											E_VERTEX_TYPE vType,
											scene::E_PRIMITIVE_TYPE pType,
											scene::E_INDEX_TYPE iType)
{
	VertexCache.vertices = vertices;
	VertexCache.vertexCount = vertexCount;

	VertexCache.indices = indices;
	VertexCache.indicesIndex = 0;
	VertexCache.indicesRun = 0;

	VertexCache.vType = vType;
	VertexCache.pType = pType;

	switch ( iType )
	{
		case EIT_16BIT: VertexCache.iType = 1; break;
		case EIT_32BIT: VertexCache.iType = 2; break;
		default:
			VertexCache.iType = iType; break;
	}

	switch ( VertexCache.pType )
	{
		// most types here will not work as expected, only triangles/triangle_fan
		// is known to work.
		case scene::EPT_POINTS:
			VertexCache.indexCount = primitiveCount;
			VertexCache.primitivePitch = 1;
			break;
		case scene::EPT_LINE_STRIP:
			VertexCache.indexCount = primitiveCount+1;
			VertexCache.primitivePitch = 1;
			break;
		case scene::EPT_LINE_LOOP:
			VertexCache.indexCount = primitiveCount+1;
			VertexCache.primitivePitch = 1;
			break;
		case scene::EPT_LINES:
			VertexCache.indexCount = 2*primitiveCount;
			VertexCache.primitivePitch = 2;
			break;
		case scene::EPT_TRIANGLE_STRIP:
			VertexCache.indexCount = primitiveCount+2;
			VertexCache.primitivePitch = 1;
			break;
		case scene::EPT_TRIANGLES:
			VertexCache.indexCount = primitiveCount + primitiveCount + primitiveCount;
			VertexCache.primitivePitch = 3;
			break;
		case scene::EPT_TRIANGLE_FAN:
			VertexCache.indexCount = primitiveCount + 2;
			VertexCache.primitivePitch = 1;
			break;
	}

	irr::memset32 ( VertexCache.info, VERTEXCACHE_MISS, sizeof ( VertexCache.info ) );
}


void CBurningVideoDriver::drawVertexPrimitiveList(const void* vertices, uint32_t vertexCount,
				const void* indexList, uint32_t primitiveCount,
				E_VERTEX_TYPE vType, scene::E_PRIMITIVE_TYPE pType, scene::E_INDEX_TYPE iType)

{
	if (!checkPrimitiveCount(primitiveCount,pType))
		return;

	CNullDriver::drawVertexPrimitiveList(vertices, vertexCount, indexList, primitiveCount, vType, pType, iType);

	// These calls would lead to crashes due to wrong index usage.
	// The vertex cache needs to be rewritten for these primitives.
	if (pType==scene::EPT_POINTS || pType==scene::EPT_LINE_STRIP ||
		pType==scene::EPT_LINE_LOOP || pType==scene::EPT_LINES)
		return;

	if ( 0 == CurrentShader )
		return;

	VertexCache_reset ( vertices, vertexCount, indexList, primitiveCount, vType, pType, iType );

	const s4DVertex * face[3];

	float dc_area;
	int32_t lodLevel;
	uint32_t i;
	uint32_t g;
	uint32_t m;
	video::CSoftwareTexture2* tex;

	for ( i = 0; i < (uint32_t) primitiveCount; ++i )
	{
		VertexCache_get(face);

		// if fully outside or outside on same side
		if ( ( (face[0]->flag | face[1]->flag | face[2]->flag) & VERTEX4D_CLIPMASK )
				!= VERTEX4D_INSIDE
			)
			continue;

		// if fully inside
		if ( ( face[0]->flag & face[1]->flag & face[2]->flag & VERTEX4D_CLIPMASK ) == VERTEX4D_INSIDE )
		{
			dc_area = screenarea2 ( face );
			if ( Material.org.BackfaceCulling && F32_LOWER_EQUAL_0( dc_area ) )
				continue;
			else
			if ( Material.org.FrontfaceCulling && F32_GREATER_EQUAL_0( dc_area ) )
				continue;

			// select mipmap
			dc_area = core::reciprocal ( dc_area );
			for ( m = 0; m != vSize[VertexCache.vType].TexSize; ++m )
			{
				if ( 0 == (tex = MAT_TEXTURE ( m )) )
				{
					CurrentShader->setTextureParam(m, 0, 0);
					continue;
				}

				lodLevel = s32_log2_f32 ( texelarea2 ( face, m ) * dc_area  );
				CurrentShader->setTextureParam(m, tex, lodLevel );
				select_polygon_mipmap2 ( (s4DVertex**) face, m, tex->getSize() );
			}

			// rasterize
			CurrentShader->drawTriangle ( face[0] + 1, face[1] + 1, face[2] + 1 );
			continue;
		}

		// else if not complete inside clipping necessary
		irr::memcpy32_small ( ( (uint8_t*) CurrentOut.data + ( 0 << ( SIZEOF_SVERTEX_LOG2 + 1 ) ) ), face[0], SIZEOF_SVERTEX * 2 );
		irr::memcpy32_small ( ( (uint8_t*) CurrentOut.data + ( 1 << ( SIZEOF_SVERTEX_LOG2 + 1 ) ) ), face[1], SIZEOF_SVERTEX * 2 );
		irr::memcpy32_small ( ( (uint8_t*) CurrentOut.data + ( 2 << ( SIZEOF_SVERTEX_LOG2 + 1 ) ) ), face[2], SIZEOF_SVERTEX * 2 );

		const uint32_t flag = CurrentOut.data->flag & VERTEX4D_FORMAT_MASK;

		for ( g = 0; g != CurrentOut.ElementSize; ++g )
		{
			CurrentOut.data[g].flag = flag;
			Temp.data[g].flag = flag;
		}

		uint32_t vOut;
		vOut = clipToFrustum ( CurrentOut.data, Temp.data, 3 );
		if ( vOut < 3 )
			continue;

		vOut <<= 1;

		// to DC Space, project homogenous vertex
		ndc_2_dc_and_project ( CurrentOut.data + 1, CurrentOut.data, vOut );

/*
		// TODO: don't stick on 32 Bit Pointer
		#define PointerAsValue(x) ( (uint32_t) (uint32_t*) (x) )

		// if not complete inside clipping necessary
		if ( ( test & VERTEX4D_INSIDE ) != VERTEX4D_INSIDE )
		{
			uint32_t v[2] = { PointerAsValue ( Temp ) , PointerAsValue ( CurrentOut ) };
			for ( g = 0; g != 6; ++g )
			{
				vOut = clipToHyperPlane ( (s4DVertex*) v[0], (s4DVertex*) v[1], vOut, NDCPlane[g] );
				if ( vOut < 3 )
					break;

				v[0] ^= v[1];
				v[1] ^= v[0];
				v[0] ^= v[1];
			}

			if ( vOut < 3 )
				continue;

		}
*/

		// check 2d backface culling on first
		dc_area = screenarea ( CurrentOut.data );
		if ( Material.org.BackfaceCulling && F32_LOWER_EQUAL_0 ( dc_area ) )
			continue;
		else if ( Material.org.FrontfaceCulling && F32_GREATER_EQUAL_0( dc_area ) )
			continue;

		// select mipmap
		dc_area = core::reciprocal ( dc_area );
		for ( m = 0; m != vSize[VertexCache.vType].TexSize; ++m )
		{
			if ( 0 == (tex = MAT_TEXTURE ( m )) )
			{
				CurrentShader->setTextureParam(m, 0, 0);
				continue;
			}

			lodLevel = s32_log2_f32 ( texelarea ( CurrentOut.data, m ) * dc_area );
			CurrentShader->setTextureParam(m, tex, lodLevel );
			select_polygon_mipmap ( CurrentOut.data, vOut, m, tex->getSize() );
		}


		// re-tesselate ( triangle-fan, 0-1-2,0-2-3.. )
		for ( g = 0; g <= vOut - 6; g += 2 )
		{
			// rasterize
			CurrentShader->drawTriangle ( CurrentOut.data + 0 + 1,
							CurrentOut.data + g + 3,
							CurrentOut.data + g + 5);
		}

	}

	// dump statistics
/*
	char buf [64];
	sprintf ( buf,"VCount:%d PCount:%d CacheMiss: %d",
					vertexCount, primitiveCount,
					VertexCache.CacheMiss
				);
	os::Printer::log( buf );
*/

}
#endif // defined


//! sets a material
void CBurningVideoDriver::setMaterial(const SGPUMaterial& material)
{
	Material.org = material;


#ifdef SOFTWARE_DRIVER_2_LIGHTING
	Material.AmbientColor.setR8G8B8 ( Material.org.AmbientColor.color );
	Material.DiffuseColor.setR8G8B8 ( Material.org.DiffuseColor.color );
	Material.EmissiveColor.setR8G8B8 ( Material.org.EmissiveColor.color );
	Material.SpecularColor.setR8G8B8 ( Material.org.SpecularColor.color );

	core::setbit_cond ( LightSpace.Flags, Material.org.Shininess != 0.f, SPECULAR );
#endif

	setCurrentShader();
}


/*!
	Camera Position in World Space
*/
void CBurningVideoDriver::getCameraPosWorldSpace ()
{
	const float *M = getTransform(E4X3TS_VIEW_INVERSE).pointer ();

	/*	The  viewpoint is at (0., 0., 0.) in eye space.
		Turning this into a vector [0 0 0 1] and multiply it by
		the inverse of the view matrix, the resulting vector is the
		object space location of the camera.
	*/

	LightSpace.campos.x = M[12];
	LightSpace.campos.y = M[13];
	LightSpace.campos.z = M[14];
	LightSpace.campos.w = 1.f;
}




//! draws an 2d image, using a color (if color is other then Color(255,255,255,255)) and the alpha channel of the texture if wanted.
void CBurningVideoDriver::draw2DImage(const video::ITexture* texture, const core::position2d<int32_t>& destPos,
					 const core::rect<int32_t>& sourceRect,
					 const core::rect<int32_t>* clipRect, SColor color,
					 bool useAlphaChannelOfTexture)
{
	if (texture)
	{
		if (texture->getDriverType() != EDT_BURNINGSVIDEO)
		{
			os::Printer::log("Fatal Error: Tried to copy from a surface not owned by this driver.", ELL_ERROR);
			return;
		}

		if (useAlphaChannelOfTexture)
			((CSoftwareTexture2*)texture)->getImage()->copyToWithAlpha(
			RenderTargetSurface, destPos, sourceRect, color, clipRect);
		else
			((CSoftwareTexture2*)texture)->getImage()->copyTo(
				RenderTargetSurface, destPos, sourceRect, clipRect);
	}
}


//! Draws a part of the texture into the rectangle.
void CBurningVideoDriver::draw2DImage(const video::ITexture* texture, const core::rect<int32_t>& destRect,
		const core::rect<int32_t>& sourceRect, const core::rect<int32_t>* clipRect,
		const video::SColor* const colors, bool useAlphaChannelOfTexture)
{
	if (texture)
	{
		if (texture->getDriverType() != EDT_BURNINGSVIDEO)
		{
			os::Printer::log("Fatal Error: Tried to copy from a surface not owned by this driver.", ELL_ERROR);
			return;
		}

	if (useAlphaChannelOfTexture)
		StretchBlit(BLITTER_TEXTURE_ALPHA_BLEND, RenderTargetSurface, &destRect, &sourceRect,
			    ((CSoftwareTexture2*)texture)->getImage(), (colors ? colors[0].color : 0));
	else
		StretchBlit(BLITTER_TEXTURE, RenderTargetSurface, &destRect, &sourceRect,
			    ((CSoftwareTexture2*)texture)->getImage(), (colors ? colors[0].color : 0));
	}
}

//! Draws a 2d line.
void CBurningVideoDriver::draw2DLine(const core::position2d<int32_t>& start,
					const core::position2d<int32_t>& end,
					SColor color)
{
	drawLine(BackBuffer, start, end, color );
}


//! Draws a pixel
void CBurningVideoDriver::drawPixel(uint32_t x, uint32_t y, const SColor & color)
{
	BackBuffer->setPixel(x, y, color, true);
}


//! draw an 2d rectangle
void CBurningVideoDriver::draw2DRectangle(SColor color, const core::rect<int32_t>& pos,
									 const core::rect<int32_t>* clip)
{
	if (clip)
	{
		core::rect<int32_t> p(pos);

		p.clipAgainst(*clip);

		if(!p.isValid())
			return;

		drawRectangle(BackBuffer, p, color);
	}
	else
	{
		if(!pos.isValid())
			return;

		drawRectangle(BackBuffer, pos, color);
	}
}


//! Only used by the internal engine. Used to notify the driver that
//! the window was resized.
void CBurningVideoDriver::OnResize(const core::dimension2d<uint32_t>& size)
{
	// make sure width and height are multiples of 2
	core::dimension2d<uint32_t> realSize(size);

	if (realSize.Width % 2)
		realSize.Width += 1;

	if (realSize.Height % 2)
		realSize.Height += 1;

	if (ScreenSize != realSize)
	{
		ScreenSize = realSize;

		bool resetRT = (RenderTargetSurface == BackBuffer);

		if (BackBuffer)
			BackBuffer->drop();
		BackBuffer = new CImage(BURNINGSHADER_COLOR_FORMAT, realSize);

		if (resetRT)
			setRenderTarget(BackBuffer);
	}
}


//! returns the current render target size
const core::dimension2d<uint32_t>& CBurningVideoDriver::getCurrentRenderTargetSize() const
{
	return RenderTargetSize;
}


//!Draws an 2d rectangle with a gradient.
void CBurningVideoDriver::draw2DRectangle(const core::rect<int32_t>& position,
	SColor colorLeftUp, SColor colorRightUp, SColor colorLeftDown, SColor colorRightDown,
	const core::rect<int32_t>* clip)
{
#ifdef SOFTWARE_DRIVER_2_USE_VERTEX_COLOR

	core::rect<int32_t> pos = position;

	if (clip)
		pos.clipAgainst(*clip);

	if (!pos.isValid())
		return;

	const core::dimension2d<int32_t> renderTargetSize ( ViewPort.getSize() );

	const int32_t xPlus = -(renderTargetSize.Width>>1);
	const float xFact = 1.0f / (renderTargetSize.Width>>1);

	const int32_t yPlus = renderTargetSize.Height-(renderTargetSize.Height>>1);
	const float yFact = 1.0f / (renderTargetSize.Height>>1);

	// fill VertexCache direct
	s4DVertex *v;
#ifndef NEW_MESHES
#error "VERTICES ARE NO LONGER WITH US"
	VertexCache.vertexCount = 4;

	VertexCache.info[0].index = 0;
	VertexCache.info[1].index = 1;
	VertexCache.info[2].index = 2;
	VertexCache.info[3].index = 3;

	v = &VertexCache.mem.data [ 0 ];

	v[0].Pos.set ( (float)(pos.UpperLeftCorner.X+xPlus) * xFact, (float)(yPlus-pos.UpperLeftCorner.Y) * yFact, 0.f, 1.f );
	v[0].Color[0].setA8R8G8B8 ( colorLeftUp.color );

	v[2].Pos.set ( (float)(pos.LowerRightCorner.X+xPlus) * xFact, (float)(yPlus- pos.UpperLeftCorner.Y) * yFact, 0.f, 1.f );
	v[2].Color[0].setA8R8G8B8 ( colorRightUp.color );

	v[4].Pos.set ( (float)(pos.LowerRightCorner.X+xPlus) * xFact, (float)(yPlus-pos.LowerRightCorner.Y) * yFact, 0.f ,1.f );
	v[4].Color[0].setA8R8G8B8 ( colorRightDown.color );

	v[6].Pos.set ( (float)(pos.UpperLeftCorner.X+xPlus) * xFact, (float)(yPlus-pos.LowerRightCorner.Y) * yFact, 0.f, 1.f );
	v[6].Color[0].setA8R8G8B8 ( colorLeftDown.color );

	int32_t i;
	uint32_t g;

	for ( i = 0; i!= 8; i += 2 )
	{
		v[i + 0].flag = clipToFrustumTest ( v + i );
		v[i + 1].flag = 0;
		if ( (v[i].flag & VERTEX4D_INSIDE ) == VERTEX4D_INSIDE )
		{
			ndc_2_dc_and_project ( v + i + 1, v + i, 2 );
		}
	}

	IBurningShader * render;

	render = BurningShader [ ETR_GOURAUD_ALPHA_NOZ ];
	render->setRenderTarget(RenderTargetSurface, ViewPort);

	static const int16_t indexList[6] = {0,1,2,0,2,3};

	s4DVertex * face[3];

	for ( i = 0; i!= 6; i += 3 )
	{
		face[0] = VertexCache_getVertex ( indexList [ i + 0 ] );
		face[1] = VertexCache_getVertex ( indexList [ i + 1 ] );
		face[2] = VertexCache_getVertex ( indexList [ i + 2 ] );

		// test clipping
		uint32_t test = face[0]->flag & face[1]->flag & face[2]->flag & VERTEX4D_INSIDE;

		if ( test == VERTEX4D_INSIDE )
		{
			render->drawTriangle ( face[0] + 1, face[1] + 1, face[2] + 1 );
			continue;
		}
		// Todo: all vertices are clipped in 2d..
		// is this true ?
		uint32_t vOut = 6;
		memcpy ( CurrentOut.data + 0, face[0], sizeof ( s4DVertex ) * 2 );
		memcpy ( CurrentOut.data + 2, face[1], sizeof ( s4DVertex ) * 2 );
		memcpy ( CurrentOut.data + 4, face[2], sizeof ( s4DVertex ) * 2 );

		vOut = clipToFrustum ( CurrentOut.data, Temp.data, 3 );
		if ( vOut < 3 )
			continue;

		vOut <<= 1;
		// to DC Space, project homogenous vertex
		ndc_2_dc_and_project ( CurrentOut.data + 1, CurrentOut.data, vOut );

		// re-tesselate ( triangle-fan, 0-1-2,0-2-3.. )
		for ( g = 0; g <= vOut - 6; g += 2 )
		{
			// rasterize
			render->drawTriangle ( CurrentOut.data + 1, &CurrentOut.data[g + 3], &CurrentOut.data[g + 5] );
		}

	}
#endif // NEW_MESHES

#else
	draw2DRectangle ( colorLeftUp, position, clip );
#endif
}



//! \return Returns the name of the video driver. Example: In case of the DirectX8
//! driver, it would return "Direct3D8.1".
const wchar_t* CBurningVideoDriver::getName() const
{
#ifdef BURNINGVIDEO_RENDERER_BEAUTIFUL
	return L"Burning's Video 0.47 beautiful";
#elif defined ( BURNINGVIDEO_RENDERER_ULTRA_FAST )
	return L"Burning's Video 0.47 ultra fast";
#elif defined ( BURNINGVIDEO_RENDERER_FAST )
	return L"Burning's Video 0.47 fast";
#else
	return L"Burning's Video 0.47";
#endif
}

//! Returns the graphics card vendor name.
std::string CBurningVideoDriver::getVendorInfo()
{
	return "Burning's Video: Ing. Thomas Alten (c) 2006-2012";
}


//! Returns type of video driver
E_DRIVER_TYPE CBurningVideoDriver::getDriverType() const
{
	return EDT_BURNINGSVIDEO;
}


//! returns color format
asset::E_FORMAT CBurningVideoDriver::getColorFormat() const
{
	return BURNINGSHADER_COLOR_FORMAT;
}

//! Clears the DepthBuffer.
void CBurningVideoDriver::clearZBuffer()
{
	if (DepthBuffer)
		DepthBuffer->clear();
}


//! Returns the maximum amount of primitives (mostly vertices) which
//! the device is able to render with one drawIndexedTriangleList
//! call.
uint32_t CBurningVideoDriver::getMaximalIndicesCount() const
{
	return 0xFFFFFFFF;
}


} // end namespace video
} // end namespace irr

#endif // _IRR_COMPILE_WITH_BURNINGSVIDEO_

namespace irr
{
namespace video
{

//! creates a video driver
IVideoDriver* createBurningVideoDriver(IrrlichtDevice* dev, const irr::SIrrlichtCreationParameters& params, io::IFileSystem* io, video::IImagePresenter* presenter)
{
	#ifdef _IRR_COMPILE_WITH_BURNINGSVIDEO_
	return new CBurningVideoDriver(dev, params, io, presenter);
	#else
	return 0;
	#endif // _IRR_COMPILE_WITH_BURNINGSVIDEO_
}



} // end namespace video
} // end namespace irr

