#ifndef _IRR_BSDF_COMMON_INCLUDED_
#define _IRR_BSDF_COMMON_INCLUDED_

#include <irr/builtin/glsl/math/constants.glsl>

#include <irr/builtin/glsl/limits/numeric.glsl>

// do not use this struct in SSBO or UBO, its wasteful on memory
struct irr_glsl_DirAndDifferential
{
   vec3 dir;
   // differentials at origin, I'd much prefer them to be differentials of barycentrics instead of position in the future
   mat2x3 dPosdScreen;
};

//TODO change name to irr_glsl_IsotropicViewSurfaceInteraction
// do not use this struct in SSBO or UBO, its wasteful on memory
struct irr_glsl_IsotropicViewSurfaceInteraction
{
   irr_glsl_DirAndDifferential V; // outgoing direction, NOT NORMALIZED; V.dir can have undef value for lambertian BSDF
   vec3 N; // surface normal, NOT NORMALIZED
   float NdotV;
   float NdotV_squared;
};
struct irr_glsl_AnisotropicViewSurfaceInteraction
{
    irr_glsl_IsotropicViewSurfaceInteraction isotropic;
    vec3 T;
    vec3 B;
    float TdotV;
    float BdotV;
};
mat3 irr_glsl_getTangentFrame(in irr_glsl_AnisotropicViewSurfaceInteraction interaction)
{
    return mat3(interaction.T,interaction.B,interaction.isotropic.N);
}

// do not use this struct in SSBO or UBO, its wasteful on memory
struct irr_glsl_BSDFSample
{
   vec3 L;  // incoming direction, normalized
   float probability; // for a single sample (don't care about number drawn)
};


// do not use this struct in SSBO or UBO, its wasteful on memory
struct irr_glsl_BSDFIsotropicParams
{
   float NdotL;
   float NdotL_squared;
   float VdotL; // same as LdotV
   float NdotH;
   float VdotH; // same as LdotH
   // left over for anisotropic calc and BSDF that want to implement fast bump mapping
   float LplusV_rcpLen;
   // basically metadata
   vec3 L;
   float invlenL2;
   irr_glsl_IsotropicViewSurfaceInteraction interaction;
};

// do not use this struct in SSBO or UBO, its wasteful on memory
struct irr_glsl_BSDFAnisotropicParams
{
   irr_glsl_BSDFIsotropicParams isotropic;
   float TdotL;
   float TdotV;
   float TdotH;
   float BdotL;
   float BdotV;
   float BdotH;
   // useless metadata
   vec3 T;
   vec3 B;
};

// chain rule on various functions (usually vertex attributes and barycentrics)
vec2 irr_glsl_applyScreenSpaceChainRule1D3(in vec3 dFdG, in mat2x3 dGdScreen)
{
   return vec2(dot(dFdG,dGdScreen[0]),dot(dFdG,dGdScreen[1]));
}
mat2 irr_glsl_applyScreenSpaceChainRule2D3(in mat3x2 dFdG, in mat2x3 dGdScreen)
{
   return dFdG*dGdScreen;
}
mat2x3 irr_glsl_applyScreenSpaceChainRule3D3(in mat3 dFdG, in mat2x3 dGdScreen)
{
   return dFdG*dGdScreen;
}
mat2x4 irr_glsl_applyScreenSpaceChainRule4D3(in mat3x4 dFdG, in mat2x3 dGdScreen)
{
   return dFdG*dGdScreen;
}

//TODO move to different glsl header
vec2 irr_glsl_concentricMapping(in vec2 _u)
{
    //map [0;1]^2 to [-1;1]^2
    vec2 u = 2.0*_u - 1.0;
    
    vec2 p;
    if (u==vec2(0.0))
        p = vec2(0.0);
    else
    {
        float r;
        float theta;
        if (abs(u.x)>abs(u.y)) {
            r = u.x;
            theta = 0.25*irr_glsl_PI * (u.y/u.x);
        } else {
            r = u.y;
            theta = 0.5*irr_glsl_PI - 0.25*irr_glsl_PI*(u.x/u.y);
        }
        p = r*vec2(cos(theta),sin(theta));
    }

    return p;
}

//TODO move this to different glsl header
mat2x3 irr_glsl_frisvad(in vec3 n)
{
	const float a = 1.0/(1.0 + n.z);
	const float b = -n.x*n.y*a;
	return (n.z<-0.9999999) ? mat2x3(vec3(0.0,-1.0,0.0),vec3(-1.0,0.0,0.0)):mat2x3(vec3(1.0-n.x*n.x*a, b, -n.x),vec3(b, 1.0-n.y*n.y*a, -n.y));
}

// only in the fragment shader we have access to implicit derivatives
irr_glsl_IsotropicViewSurfaceInteraction irr_glsl_calcFragmentShaderSurfaceInteraction(in vec3 _CamPos, in vec3 _SurfacePos, in vec3 _Normal)
{
   irr_glsl_IsotropicViewSurfaceInteraction interaction;
   interaction.V.dir = _CamPos-_SurfacePos;
   interaction.V.dPosdScreen[0] = dFdx(_SurfacePos);
   interaction.V.dPosdScreen[1] = dFdy(_SurfacePos);
   interaction.N = _Normal;
   float invlenV2 = inversesqrt(dot(interaction.V.dir,interaction.V.dir));
   float invlenN2 = inversesqrt(dot(interaction.N,interaction.N));
   interaction.V.dir *= invlenV2;
   interaction.N *= invlenN2;
   interaction.NdotV = dot(interaction.N,interaction.V.dir);
   interaction.NdotV_squared = interaction.NdotV*interaction.NdotV;
   return interaction;
}
irr_glsl_AnisotropicViewSurfaceInteraction irr_glsl_calcAnisotropicInteraction(in irr_glsl_IsotropicViewSurfaceInteraction isotropic, in vec3 T, in vec3 B)
{
    irr_glsl_AnisotropicViewSurfaceInteraction inter;
    inter.isotropic = isotropic;
    inter.T = T;
    inter.B = B;
    inter.TdotV = dot(inter.isotropic.V.dir,inter.T);
    inter.BdotV = dot(inter.isotropic.V.dir,inter.B);

    return inter;
}
irr_glsl_AnisotropicViewSurfaceInteraction irr_glsl_calcAnisotropicInteraction(in irr_glsl_IsotropicViewSurfaceInteraction isotropic, in vec3 T)
{
    return irr_glsl_calcAnisotropicInteraction(isotropic, T, cross(isotropic.N,T));
}
irr_glsl_AnisotropicViewSurfaceInteraction irr_glsl_calcAnisotropicInteraction(in irr_glsl_IsotropicViewSurfaceInteraction isotropic)
{
    mat2x3 TB = irr_glsl_frisvad(isotropic.N);
    return irr_glsl_calcAnisotropicInteraction(isotropic, TB[0], TB[1]);
}
/*
//TODO it doesnt compile, lots of undefined symbols
// when you know the projected positions of your triangles (TODO: should probably make a function like this that also computes barycentrics)
irr_glsl_IsotropicViewSurfaceInteraction irr_glsl_calcBarycentricSurfaceInteraction(in vec3 _CamPos, in vec3 _SurfacePos[3], in vec3 _Normal[3], in float _Barycentrics[2], in vec2 _ProjectedPos[3])
{
   irr_glsl_IsotropicViewSurfaceInteraction interaction;

   // Barycentric interpolation = b0*attr0+b1*attr1+attr2*(1-b0-b1)
   vec3 b = vec3(_Barycentrics[0],_Barycentrics[1],1.0-_Barycentrics[0]-_Barycentrics[1]);
   mat3 vertexAttrMatrix = mat3(_SurfacePos[0],_SurfacePos[1],_SurfacePos[2]);
   interaction.V.dir = _CamPos-vertexAttrMatrix*b;
   // Schied's derivation - modified
   vec2 to2 = _ProjectedPos[2]-_ProjectedPos[1];
   vec2 to1 = _ProjectedPos[0]-_ProjectedPos[1];
   float d = 1.0/determinant(mat2(to2,to1)); // TODO double check all this
   mat2x3 dBaryd = mat2x3(vec3(v[1].y-v[2].y,to2.y,to0.y)*d,-vec3(v[1].x-v[2].x,to2.x,t0.x)*d);
   //
   interaction.dPosdScreen = irr_glsl_applyScreenSpaceChainRule3D3(vertexAttrMatrix,dBaryd);

   vertexAttrMatrix = mat3(_Normal[0],_Normal[1],_Normal[2]);
   interaction.N = vertexAttrMatrix*b;

   return interaction;
}
// when you know the ray and triangle it hits
irr_glsl_IsotropicViewSurfaceInteraction  irr_glsl_calcRaySurfaceInteraction(in irr_glsl_DirAndDifferential _rayTowardsSurface, in vec3 _SurfacePos[3], in vec3 _Normal[3], in float _Barycentrics[2])
{
   irr_glsl_IsotropicViewSurfaceInteraction interaction;
   // flip ray
   interaction.V.dir = -_rayTowardsSurface.dir;
   // do some hardcore shizz to transform a differential at origin into a differential at surface
   // also in barycentrics preferably (turn world pos diff into bary diff with applyScreenSpaceChainRule3D3)
   //interaction.V.dPosdx = TODO;
   //interaction.V.dPosdy = TODO;

   vertexAttrMatrix = mat3(_Normal[0],_Normal[1],_Normal[2]);
   interaction.N = vertexAttrMatrix*b;

   return interaction;
}
*/

// will normalize all the vectors
irr_glsl_BSDFIsotropicParams irr_glsl_calcBSDFIsotropicParams(in irr_glsl_IsotropicViewSurfaceInteraction interaction, in vec3 L)
{
   float invlenL2 = inversesqrt(dot(L,L));

   irr_glsl_BSDFIsotropicParams params;

   // totally useless vectors, will probably get optimized away by compiler if they don't get used
   // but useful as temporaries
   params.interaction = interaction;
   params.L = L*invlenL2;
   params.invlenL2 = invlenL2;

   // this stuff only works with normalized L,N,V
   params.NdotL = dot(params.interaction.N,params.L);
   params.NdotL_squared = params.NdotL*params.NdotL;

   params.VdotL = dot(params.interaction.V.dir,params.L);
   float LplusV_rcpLen = inversesqrt(2.0 + 2.0*params.VdotL);
   params.LplusV_rcpLen = LplusV_rcpLen;

   // this stuff works unnormalized L,N,V
   params.NdotH = (params.NdotL+params.interaction.NdotV)*LplusV_rcpLen;
   params.VdotH = LplusV_rcpLen + LplusV_rcpLen*params.VdotL;

   return params;
}
// get extra stuff for anisotropy, here we actually require T and B to be normalized
irr_glsl_BSDFAnisotropicParams irr_glsl_calcBSDFAnisotropicParams(in irr_glsl_BSDFIsotropicParams isotropic, in vec3 T, in vec3 B)
{
   irr_glsl_BSDFAnisotropicParams params;
   params.isotropic = isotropic;

   // meat
   params.TdotL = dot(T,isotropic.L);
   params.TdotV = dot(T,isotropic.interaction.V.dir);
   params.TdotH = (params.TdotV+params.TdotL)*isotropic.LplusV_rcpLen;
   params.BdotL = dot(B,isotropic.L);
   params.BdotV = dot(B,isotropic.interaction.V.dir);
   params.BdotH = (params.BdotV+params.BdotL)*isotropic.LplusV_rcpLen;

   // useless stuff we keep just to be complete
   params.T = T;
   params.B = B;

   return params;
}
#endif
