/* irrlicht.h -- interface of the 'Irrlicht Engine'

  Copyright (C) 2002-2012 Nikolaus Gebhardt

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  Please note that the Irrlicht Engine is based in part on the work of the
  Independent JPEG Group, the zlib and the libPng. This means that if you use
  the Irrlicht Engine in your product, you must acknowledge somewhere in your
  documentation that you've used the IJG code. It would also be nice to mention
  that you use the Irrlicht Engine, the zlib and libPng. See the README files
  in the jpeglib, the zlib and libPng for further informations.
*/

#ifndef __IRRLICHT_H_INCLUDED__
#define __IRRLICHT_H_INCLUDED__

// overarching includes
#include "irr/core/BaseClasses.h"
#include "irr/macros.h"

//core
#include "irr/core/IReferenceCounted.h"
#include "irr/core/Types.h"
#include "irr/core/irrString.h"
#include "irr/core/math/irrMath.h"


#include "IrrCompileConfig.h"
#include "aabbox3d.h"
#include "coreutil.h"
#include "dimension2d.h"
#include "CFinalBoneHierarchy.h"
#include "CGLSLFunctionGenerator.h"
#include "ECullingTypes.h"
#include "EDebugSceneTypes.h"
#include "EDriverTypes.h"
#include "EMaterialFlags.h"
#include "EMaterialTypes.h"
#include "EMeshWriterEnums.h"
#include "ESceneNodeAnimatorTypes.h"
#include "ESceneNodeTypes.h"
#include "IAnimatedMesh.h"
#include "IAnimatedMeshSceneNode.h"
#include "IBillboardSceneNode.h"
#include "ICameraSceneNode.h"
#include "ICursorControl.h"
#include "IDummyTransformationSceneNode.h"
#include "IEventReceiver.h"
#include "IFileList.h"
#include "IFileSystem.h"
#include "IGeometryCreator.h"
#include "IGPUTransientBuffer.h"
#include "IGPUProgrammingServices.h"
#include "CImageData.h"
#include "IImage.h"
#include "IImageLoader.h"
#include "IImageWriter.h"
#include "ILogger.h"
#include "IMaterialRenderer.h"
#include "IMaterialRendererServices.h"
#include "IMesh.h"
#include "IMeshBuffer.h"
#include "IMeshCache.h"
#include "IMeshLoader.h"
#include "IMeshManipulator.h"
#include "IMeshSceneNode.h"
#include "IMeshSceneNodeInstanced.h"
#include "IMeshWriter.h"
#include "IMetaGranularBuffer.h"
#include "IOSOperator.h"
#include "IReadFile.h"
#include "IrrlichtDevice.h"
#include "path.h"
#include "ISceneManager.h"
#include "ISceneNode.h"
#include "ISceneNodeAnimator.h"
#include "ISceneNodeAnimatorCameraFPS.h"
#include "ISceneNodeAnimatorCameraMaya.h"
#include "IShaderConstantSetCallBack.h"
#include "ISkinnedMesh.h"
#include "ISkinnedMeshSceneNode.h"
#include "ITexture.h"
#include "ITextureBufferObject.h"
#include "IMultisampleTexture.h"
#include "ITimer.h"
#include "ITransformFeedback.h"
#include "IVideoDriver.h"
#include "IVideoModeList.h"
#include "IWriteFile.h"
#include "Keycodes.h"
#include "line2d.h"
#include "line3d.h"
#include "matrix4.h"
#include "matrixSIMD4.h"
#include "plane3d.h"
#include "position2d.h"
#include "quaternion.h"
#include "rect.h"
#include "splines.h"


#include "SceneParameters.h"
#include "SColor.h"
#include "SCollisionEngine.h"
#include "SExposedVideoData.h"
#include "SIrrCreationParameters.h"
#include "SKeyMap.h"
#include "SMaterial.h"
#include "SMesh.h"
#include "SSkinMeshBuffer.h"
#include "SViewFrustum.h"
#include "triangle3d.h"
#include "vector2d.h"
#include "vector3d.h"
#include "vectorSIMD.h"


#include "SIrrCreationParameters.h"

//! Everything in the Irrlicht Engine can be found in this namespace.
namespace irr
{
	//! Creates an Irrlicht device. The Irrlicht device is the root object for using the engine.
	/** If you need more parameters to be passed to the creation of the Irrlicht Engine device,
	use the createDeviceEx() function.
	\param deviceType: Type of the device. This can currently be video::EDT_NULL,
	video::EDT_BURNINGSVIDEO, and video::EDT_OPENGL.
	\param windowSize: Size of the window or the video mode in fullscreen mode.
	\param bits: Bits per pixel in fullscreen mode. Ignored if windowed mode.
	\param fullscreen: Should be set to true if the device should run in fullscreen. Otherwise
		the device runs in windowed mode.
	\param stencilbuffer: Specifies if the stencil buffer should be enabled. Set this to true,
	if you want the engine be able to draw stencil buffer shadows. Note that not all
	devices are able to use the stencil buffer. If they don't no shadows will be drawn.
	\param vsync: Specifies vertical syncronisation: If set to true, the driver will wait
	for the vertical retrace period, otherwise not.
	\param receiver: A user created event receiver.
	\return Returns pointer to the created IrrlichtDevice or null if the
	device could not be created.
	*/
	extern "C" IRRLICHT_API IrrlichtDevice* IRRCALLCONV createDevice(
		video::E_DRIVER_TYPE deviceType = video::EDT_BURNINGSVIDEO,
		// parantheses are necessary for some compilers
		const core::dimension2d<uint32_t>& windowSize = (core::dimension2d<uint32_t>(640,480)),
		uint32_t bits = 16,
		bool fullscreen = false,
		bool stencilbuffer = false,
		bool vsync = false,
		IEventReceiver* receiver = 0);

	//! typedef for Function Pointer
	typedef IrrlichtDevice* (IRRCALLCONV *funcptr_createDevice )(
			video::E_DRIVER_TYPE deviceType,
			const core::dimension2d<uint32_t>& windowSize,
			uint32_t bits,
			bool fullscreen,
			bool stencilbuffer,
			bool vsync,
			IEventReceiver* receiver);


	//! Creates an Irrlicht device with the option to specify advanced parameters.
	/** Usually you should used createDevice() for creating an Irrlicht Engine device.
	Use this function only if you wish to specify advanced parameters like a window
	handle in which the device should be created.
	\param parameters: Structure containing advanced parameters for the creation of the device.
	See irr::SIrrlichtCreationParameters for details.
	\return Returns pointer to the created IrrlichtDevice or null if the
	device could not be created. */
	extern "C" IRRLICHT_API IrrlichtDevice* IRRCALLCONV createDeviceEx(
		const SIrrlichtCreationParameters& parameters);

	//! typedef for Function Pointer
	typedef IrrlichtDevice* (IRRCALLCONV *funcptr_createDeviceEx )( const SIrrlichtCreationParameters& parameters );


	// THE FOLLOWING IS AN EMPTY LIST OF ALL SUB NAMESPACES
	// EXISTING ONLY FOR THE DOCUMENTATION SOFTWARE DOXYGEN.

	//! Basic classes such as vectors, planes, arrays, lists, and so on can be found in this namespace.
	namespace core
	{
	}

	//! The gui namespace contains useful classes for easy creation of a graphical user interface.
	namespace gui
	{
	}

	//! This namespace provides interfaces for input/output: Reading and writing files, accessing zip archives, xml files, ...
	namespace io
	{
	}

	//! All scene management can be found in this namespace: Mesh loading, special scene nodes like octrees and billboards, ...
	namespace scene
	{
	}

	//! The video namespace contains classes for accessing the video driver. All 2d and 3d rendering is done here.
	namespace video
	{
	}
}

/*! \file irrlicht.h
	\brief Main header file of the irrlicht, the only file needed to include.
*/

#endif

