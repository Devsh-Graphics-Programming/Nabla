// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#if 0

//extra extension name that is reported as supported when irrbaw app is running in renderdoc
_NBL_STATIC_INLINE_CONSTEXPR const char* RUNNING_IN_RENDERDOC_EXTENSION_NAME = "GL_NBL_RUNNING_IN_RENDERDOC";

	//! Maxmimum texture layers supported by the engine
	static uint8_t MaxTextureUnits;
	//! Maximal Anisotropy
	static uint8_t MaxAnisotropy;
	//! Number of user clipplanes
	static uint8_t MaxUserClipPlanes;
	//! Number of rendertargets available as MRTs
	static uint8_t MaxMultipleRenderTargets;
	//! Optimal number of indices per meshbuffer
	static uint32_t MaxIndices;
	//! Optimal number of vertices per meshbuffer
	static uint32_t MaxVertices;
	//! Maximal vertices handled by geometry shaders
	static uint32_t MaxGeometryVerticesOut;
	//! Maximal LOD Bias
	static float MaxTextureLODBias;
	//!
	static uint32_t MaxVertexStreams;
	//!
	static uint32_t MaxXFormFeedbackComponents;
	//!
	static uint32_t MaxGPUWaitTimeout;
	//! Gives the upper and lower bound on warp/wavefront/SIMD-lane size
	static uint32_t InvocationSubGroupSize[2];

    //TODO should be later changed to SPIR-V extensions enum like it is with OpenGL extensions
    //(however it does not have any implications on API)
    static GLuint SPIR_VextensionsCount;
    static core::smart_refctd_dynamic_array<const GLubyte*> SPIR_Vextensions;

	//! Minimal and maximal supported thickness for lines without smoothing
	GLfloat DimAliasedLine[2];
	//! Minimal and maximal supported thickness for points without smoothing
	GLfloat DimAliasedPoint[2];
	//! Minimal and maximal supported thickness for lines with smoothing
	GLfloat DimSmoothedLine[2];
	//! Minimal and maximal supported thickness for points with smoothing
	GLfloat DimSmoothedPoint[2];



}
}

#endif

#endif

