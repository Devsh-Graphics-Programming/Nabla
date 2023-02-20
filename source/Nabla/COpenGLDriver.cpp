// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#ifdef _NBL_COMPILE_WITH_OPENGL_



// -----------------------------------------------------------------------
// METHODS
// -----------------------------------------------------------------------

uint16_t COpenGLDriver::retrieveDisplayRefreshRate() const
{
#if defined(_NBL_COMPILE_WITH_WINDOWS_DEVICE_)
    DEVMODEA dm;
    dm.dmSize = sizeof(DEVMODE);
    dm.dmDriverExtra = 0;
    if (!EnumDisplaySettings(NULL, ENUM_CURRENT_SETTINGS, &dm))
        return 0u;
    return static_cast<uint16_t>(dm.dmDisplayFrequency);
#elif defined(_NBL_COMPILE_WITH_X11_DEVICE_)
#   ifdef _NBL_LINUX_X11_RANDR_
    Display* disp = XOpenDisplay(NULL);
    Window root = RootWindow(disp, 0);

    XRRScreenConfiguration* conf = XRRGetScreenInfo(disp, root);
    uint16_t rate = XRRConfigCurrentRate(conf);

    return rate;
#   else
#       ifdef _NBL_DEBUG
    os::Printer::log("Refresh rate retrieval without Xrandr compiled in is not supprted!\n", ELL_WARNING);
#       endif
    return 0u;
#   endif // _NBL_LINUX_X11_RANDR_
#else
    return 0u;
#endif
}



bool COpenGLDriver::genericDriverInit(asset::IAssetManager* assMgr)
{
	if (!AuxContexts) // opengl dead and never inited
		return false;

#ifdef _NBL_WINDOWS_API_
    if (GetModuleHandleA("renderdoc.dll"))
#elif defined(_NBL_ANDROID_PLATFORM_)
    if (dlopen("libVkLayer_GLES_RenderDoc.so", RTLD_NOW | RTLD_NOLOAD))
#elif defined(_NBL_LINUX_PLATFORM_)
    if (dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD))
#else
    if (false)
#endif
        runningInRenderDoc = true;

	Name=L"OpenGL ";
	Name.append(glGetString(GL_VERSION));
	int32_t pos=Name.findNext(L' ', 7);
	if (pos != -1)
		Name=Name.subString(0, pos);

	// print renderer information
	const GLubyte* renderer = glGetString(GL_RENDERER);
	const GLubyte* vendor = glGetString(GL_VENDOR);
	if (renderer && vendor)
	{
		os::Printer::log(reinterpret_cast<const char*>(renderer), reinterpret_cast<const char*>(vendor), ELL_INFORMATION);
		VendorName = reinterpret_cast<const char*>(vendor);
	}

    // TODO: disregard those
    maxConcurrentShaderInvocations = 4;
    maxALUShaderInvocations = 4;
    maxShaderComputeUnits = 1;


	GLint num = 0;
	glGetIntegerv(GL_MAX_TEXTURE_SIZE, &num);
	MaxTextureSizes[IGPUImageView::ET_1D][0] = static_cast<uint32_t>(num);
	MaxTextureSizes[IGPUImageView::ET_2D][0] = static_cast<uint32_t>(num);
	MaxTextureSizes[IGPUImageView::ET_2D][1] = static_cast<uint32_t>(num);

	MaxTextureSizes[IGPUImageView::ET_1D_ARRAY][0] = static_cast<uint32_t>(num);
	MaxTextureSizes[IGPUImageView::ET_2D_ARRAY][0] = static_cast<uint32_t>(num);
	MaxTextureSizes[IGPUImageView::ET_2D_ARRAY][1] = static_cast<uint32_t>(num);

	glGetIntegerv(GL_MAX_CUBE_MAP_TEXTURE_SIZE , &num);
	MaxTextureSizes[IGPUImageView::ET_CUBE_MAP][0] = static_cast<uint32_t>(num);
	MaxTextureSizes[IGPUImageView::ET_CUBE_MAP][1] = static_cast<uint32_t>(num);

	MaxTextureSizes[IGPUImageView::ET_CUBE_MAP_ARRAY][0] = static_cast<uint32_t>(num);
	MaxTextureSizes[IGPUImageView::ET_CUBE_MAP_ARRAY][1] = static_cast<uint32_t>(num);

	glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &num);
	MaxTextureSizes[IGPUImageView::ET_1D_ARRAY][2] = static_cast<uint32_t>(num);
	MaxTextureSizes[IGPUImageView::ET_2D_ARRAY][2] = static_cast<uint32_t>(num);
	MaxTextureSizes[IGPUImageView::ET_CUBE_MAP_ARRAY][2] = static_cast<uint32_t>(num);

	glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &num);
	MaxTextureSizes[IGPUImageView::ET_3D][0] = static_cast<uint32_t>(num);
	MaxTextureSizes[IGPUImageView::ET_3D][1] = static_cast<uint32_t>(num);
	MaxTextureSizes[IGPUImageView::ET_3D][2] = static_cast<uint32_t>(num);


	glGetIntegerv(GL_MAX_TEXTURE_BUFFER_SIZE , &num);
	///MaxBufferViewSize = static_cast<uint32_t>(num);


	// load extensions
	initExtensions(Params.Stencilbuffer);

    char buf[32];
    const uint32_t maj = ShaderLanguageVersion/10;
    snprintf(buf, 32, "%u.%u", maj, ShaderLanguageVersion-maj*10);
    os::Printer::log("GLSL version", buf, ELL_INFORMATION);

    if (Version<430)
    {
		os::Printer::log("OpenGL version is less than 4.3", ELL_ERROR);
		return false;
    }

	// adjust provoking vertex to match Vulkan
	extGlProvokingVertex(GL_FIRST_VERTEX_CONVENTION_EXT);

	// We need to reset once more at the beginning of the first rendering.
	// This fixes problems with intermediate changes to the material during texture load.
    SAuxContext* found = getThreadContext_helper(false);
	extGlClipControl(GL_UPPER_LEFT, GL_ZERO_TO_ONE); //once set and should never change (engine doesnt track it)
    glEnable(GL_FRAMEBUFFER_SRGB);//once set and should never change (engine doesnt track it)
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);//once set and should never change (engine doesnt track it)
    glDepthRange(1.0, 0.0);//once set and should never change (engine doesnt track it)
    found->nextState.rasterParams.multisampleEnable = 0;
    found->nextState.rasterParams.depthFunc = GL_GEQUAL;
    found->nextState.rasterParams.frontFace = GL_CCW;

	return CNullDriver::genericDriverInit(assMgr);
}



const core::smart_refctd_dynamic_array<std::string> COpenGLDriver::getSupportedGLSLExtensions() const
{
    constexpr size_t GLSLcnt = std::extent<decltype(m_GLSLExtensions)>::value;
    if (!m_supportedGLSLExtsNames)
    {
        size_t cnt = 0ull;
        for (size_t i = 0ull; i < GLSLcnt; ++i)
            cnt += (FeatureAvailable[m_GLSLExtensions[i]]);
        if (runningInRenderDoc)
            ++cnt;
        m_supportedGLSLExtsNames = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<std::string>>(cnt);
        size_t i = 0ull;
        for (size_t j = 0ull; j < GLSLcnt; ++j)
            if (FeatureAvailable[m_GLSLExtensions[j]])
                (*m_supportedGLSLExtsNames)[i++] = OpenGLFeatureStrings[m_GLSLExtensions[j]];
        if (runningInRenderDoc)
            (*m_supportedGLSLExtsNames)[i] = RUNNING_IN_RENDERDOC_EXTENSION_NAME;
    }

    return m_supportedGLSLExtsNames;
}

#endif // _NBL_COMPILE_WITH_OPENGL_
