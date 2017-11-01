// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __S_MATERIAL_H_INCLUDED__
#define __S_MATERIAL_H_INCLUDED__

#include "SColor.h"
#include "matrix4.h"
#include "irrArray.h"
#include "irrMath.h"
#include "EMaterialTypes.h"
#include "EMaterialFlags.h"
#include "SMaterialLayer.h"

namespace irr
{
namespace video
{
	class ITexture;


	//! Values defining the blend operation used when blend is enabled
	enum E_BLEND_OPERATION
	{
		EBO_NONE = 0,	//!< No blending happens
		EBO_ADD,		//!< Default blending adds the color values
		EBO_SUBTRACT,	//!< This mode subtracts the color values
		EBO_REVSUBTRACT,//!< This modes subtracts destination from source
		EBO_MIN,		//!< Choose minimum value of each color channel
		EBO_MAX,		//!< Choose maximum value of each color channel
		EBO_MIN_FACTOR,	//!< Choose minimum value of each color channel after applying blend factors, not widely supported
		EBO_MAX_FACTOR,	//!< Choose maximum value of each color channel after applying blend factors, not widely supported
		EBO_MIN_ALPHA,	//!< Choose minimum value of each color channel based on alpha value, not widely supported
		EBO_MAX_ALPHA	//!< Choose maximum value of each color channel based on alpha value, not widely supported
	};

	//! Comparison function, e.g. for depth buffer test
	enum E_COMPARISON_FUNC
	{
		//! Test never succeeds, this equals disable
		ECFN_NEVER=0,
		//! <= test
		ECFN_LESSEQUAL=1,
		//! Exact equality
		ECFN_EQUAL=2,
		//! exclusive less comparison, i.e. <
		ECFN_LESS,
		//! Succeeds almost always, except for exact equality
		ECFN_NOTEQUAL,
		//! >= test, default for e.g. depth test
		ECFN_GREATEREQUAL,
		//! inverse of <=
		ECFN_GREATER,
		//! test succeeds always
		ECFN_ALWAYS
	};

	//! Enum values for enabling/disabling color planes for rendering
	enum E_COLOR_PLANE
	{
		//! No color enabled
		ECP_NONE=0,
		//! Alpha enabled
		ECP_ALPHA=1,
		//! Red enabled
		ECP_RED=2,
		//! Green enabled
		ECP_GREEN=4,
		//! Blue enabled
		ECP_BLUE=8,
		//! All colors, no alpha
		ECP_RGB=14,
		//! All planes enabled
		ECP_ALL=15
	};



	//! Maximum number of texture an SMaterial can have.
	const uint32_t MATERIAL_MAX_TEXTURES = _IRR_MATERIAL_MAX_TEXTURES_;

	//! Struct for holding parameters for a material renderer
	class SMaterial
	{
	public:
		//! Default constructor. Creates a solid, lit material with white colors
		SMaterial()
		: MaterialType(EMT_SOLID), AmbientColor(255,255,255,255), DiffuseColor(255,255,255,255),
			EmissiveColor(0,0,0,0), SpecularColor(255,255,255,255),
			Shininess(0.0f), MaterialTypeParam(0.0f), MaterialTypeParam2(0.0f), userData(NULL), Thickness(1.0f),
			ZBuffer(ECFN_GREATEREQUAL), ColorMask(ECP_ALL),
			BlendOperation(EBO_NONE),
			PolygonOffsetConstantMultiplier(0.f), PolygonOffsetGradientMultiplier(0.f),
			Wireframe(false), PointCloud(false), ZWriteEnable(true), BackfaceCulling(true), FrontfaceCulling(false), RasterizerDiscard(false)
		{ }

		//! Copy constructor
		/** \param other Material to copy from. */
		SMaterial(const SMaterial& other)
		{
			*this = other;
		}

		//! Assignment operator
		/** \param other Material to copy from. */
		SMaterial& operator=(const SMaterial& other)
		{
			// Check for self-assignment!
			if (this == &other)
				return *this;

			MaterialType = other.MaterialType;

			AmbientColor = other.AmbientColor;
			DiffuseColor = other.DiffuseColor;
			EmissiveColor = other.EmissiveColor;
			SpecularColor = other.SpecularColor;
			Shininess = other.Shininess;
			MaterialTypeParam = other.MaterialTypeParam;
			MaterialTypeParam2 = other.MaterialTypeParam2;
			userData = other.userData;
			Thickness = other.Thickness;
			for (uint32_t i=0; i<MATERIAL_MAX_TEXTURES; ++i)
			{
				TextureLayer[i] = other.TextureLayer[i];
			}

			Wireframe = other.Wireframe;
			PointCloud = other.PointCloud;
			ZWriteEnable = other.ZWriteEnable;
			BackfaceCulling = other.BackfaceCulling;
			FrontfaceCulling = other.FrontfaceCulling;
			RasterizerDiscard = other.RasterizerDiscard;
			ZBuffer = other.ZBuffer;
			ColorMask = other.ColorMask;
			BlendOperation = other.BlendOperation;
			PolygonOffsetConstantMultiplier = other.PolygonOffsetConstantMultiplier;
			PolygonOffsetGradientMultiplier = other.PolygonOffsetGradientMultiplier;

			return *this;
		}

		//! Texture layer array.
		SMaterialLayer TextureLayer[MATERIAL_MAX_TEXTURES];

		//! Type of the material. Specifies how everything is blended together
		E_MATERIAL_TYPE MaterialType;

		//! How much ambient light (a global light) is reflected by this material.
		/** The default is full white, meaning objects are completely
		globally illuminated. Reduce this if you want to see diffuse
		or specular light effects. */
		SColor AmbientColor;

		//! How much diffuse light coming from a light source is reflected by this material.
		/** The default is full white. */
		SColor DiffuseColor;

		//! Light emitted by this material. Default is to emit no light.
		SColor EmissiveColor;

		//! How much specular light (highlights from a light) is reflected.
		/** The default is to reflect white specular light. See
		SMaterial::Shininess on how to enable specular lights. */
		SColor SpecularColor;

		//! Value affecting the size of specular highlights.
		/** A value of 20 is common. If set to 0, no specular
		highlights are being used. To activate, simply set the
		shininess of a material to a value in the range [0.5;128]:
		\code
		sceneNode->getMaterial(0).Shininess = 20.0f;
		\endcode

		You can change the color of the highlights using
		\code
		sceneNode->getMaterial(0).SpecularColor.set(255,255,255,255);
		\endcode

		The specular color of the dynamic lights
		(SLight::SpecularColor) will influence the the highlight color
		too, but they are set to a useful value by default when
		creating the light scene node. Here is a simple example on how
		to use specular highlights:
		\code
		// load and display mesh
		scene::IAnimatedMeshSceneNode* node = smgr->addAnimatedMeshSceneNode(
		smgr->getMesh("data/faerie.md2"));
		node->setMaterialTexture(0, driver->getTexture("data/Faerie2.pcx")); // set diffuse texture
		node->setMaterialFlag(video::EMF_LIGHTING, true); // enable dynamic lighting
		node->getMaterial(0).Shininess = 20.0f; // set size of specular highlights
		\endcode */
		float Shininess;

		//! Free parameter, dependent on the material type.
		/** Mostly ignored, used for example in EMT_PARALLAX_MAP_SOLID
		and EMT_TRANSPARENT_ALPHA_CHANNEL. */
		float MaterialTypeParam;

		//! Second free parameter, dependent on the material type.
		/** Mostly ignored. */
		float MaterialTypeParam2;

		//! User Data
		void* userData;
		int32_t userData2_irrmat;			// sodan (this one is mine, MINE, MINE!!!)
		int32_t userData2_textureatlas;		// sodan (this one is mine, MINE, MINE!!!)

		//! Thickness of non-3dimensional elements such as lines and points.
		float Thickness;

		//! Is the ZBuffer enabled? Default: ECFN_GREATEREQUAL
		/** Values are from E_COMPARISON_FUNC. */
		uint8_t ZBuffer;

		//! Defines the enabled color planes
		/** Values are defined as or'ed values of the E_COLOR_PLANE enum.
		Only enabled color planes will be rendered to the current render
		target. Typical use is to disable all colors when rendering only to
		depth or stencil buffer, or using Red and Green for Stereo rendering. */
		uint8_t ColorMask:4;

		//! Store the blend operation of choice
		/** Values to be chosen from E_BLEND_OPERATION. The actual way to use this value
		is not yet determined, so ignore it for now. */
		E_BLEND_OPERATION BlendOperation:4;

		float PolygonOffsetConstantMultiplier;

		float PolygonOffsetGradientMultiplier;

		//! Draw as wireframe or filled triangles? Default: false
		/** The user can access a material flag using
		\code material.Wireframe=true \endcode
		or \code material.setFlag(EMF_WIREFRAME, true); \endcode */
		bool Wireframe:1;

		//! Draw as point cloud or filled triangles? Default: false
		bool PointCloud:1;

		//! Is the zbuffer writeable or is it read-only. Default: true.
		/** This flag is forced to false if the MaterialType is a
		transparent type and the scene parameter
		ALLOW_ZWRITE_ON_TRANSPARENT is not set. */
		bool ZWriteEnable:1;

		//! Is backface culling enabled? Default: true
		bool BackfaceCulling:1;

		//! Is frontface culling enabled? Default: false
		bool FrontfaceCulling:1;

		bool RasterizerDiscard:1;

		//! Gets the i-th texture
		/** \param i The desired level.
		\return Texture for texture level i, if defined, else 0. */
		ITexture* getTexture(uint32_t i) const
		{
			return i < MATERIAL_MAX_TEXTURES ? TextureLayer[i].Texture : 0;
		}

		//! Sets the i-th texture
		/** If i>=MATERIAL_MAX_TEXTURES this setting will be ignored.
		\param i The desired level.
		\param tex Texture for texture level i. */
		void setTexture(uint32_t i, ITexture* tex)
		{
			if (i>=MATERIAL_MAX_TEXTURES)
				return;
			TextureLayer[i].Texture = tex;
		}

		//! Sets the Material flag to the given value
		/** \param flag The flag to be set.
		\param value The new value for the flag. */
		void setFlag(E_MATERIAL_FLAG flag, bool value)
		{
			switch (flag)
			{
				case EMF_WIREFRAME:
					Wireframe = value; break;
				case EMF_POINTCLOUD:
					PointCloud = value; break;
				case EMF_ZBUFFER:
					ZBuffer = value; break;
				case EMF_ZWRITE_ENABLE:
					ZWriteEnable = value; break;
				case EMF_BACK_FACE_CULLING:
					BackfaceCulling = value; break;
				case EMF_FRONT_FACE_CULLING:
					FrontfaceCulling = value; break;
				case EMF_COLOR_MASK:
					ColorMask = value?ECP_ALL:ECP_NONE; break;
				case EMF_BLEND_OPERATION:
					BlendOperation = value?EBO_ADD:EBO_NONE; break;
				default:
					break;
			}
		}

		//! Gets the Material flag
		/** \param flag The flag to query.
		\return The current value of the flag. */
		bool getFlag(E_MATERIAL_FLAG flag) const
		{
			switch (flag)
			{
				case EMF_WIREFRAME:
					return Wireframe;
				case EMF_POINTCLOUD:
					return PointCloud;
				case EMF_ZBUFFER:
					return ZBuffer!=ECFN_NEVER;
				case EMF_ZWRITE_ENABLE:
					return ZWriteEnable;
				case EMF_BACK_FACE_CULLING:
					return BackfaceCulling;
				case EMF_FRONT_FACE_CULLING:
					return FrontfaceCulling;
				case EMF_COLOR_MASK:
					return (ColorMask!=ECP_NONE);
				case EMF_BLEND_OPERATION:
					return BlendOperation != EBO_NONE;
			}

			return false;
		}

		//! Inequality operator
		/** \param b Material to compare to.
		\return True if the materials differ, else false. */
		inline bool operator!=(const SMaterial& b) const
		{
			bool different =
				MaterialType != b.MaterialType ||
				AmbientColor != b.AmbientColor ||
				DiffuseColor != b.DiffuseColor ||
				EmissiveColor != b.EmissiveColor ||
				SpecularColor != b.SpecularColor ||
				Shininess != b.Shininess ||
				MaterialTypeParam != b.MaterialTypeParam ||
				MaterialTypeParam2 != b.MaterialTypeParam2 ||
				userData != b.userData ||
				Thickness != b.Thickness ||
				Wireframe != b.Wireframe ||
				PointCloud != b.PointCloud ||
				ZBuffer != b.ZBuffer ||
				ZWriteEnable != b.ZWriteEnable ||
				BackfaceCulling != b.BackfaceCulling ||
				FrontfaceCulling != b.FrontfaceCulling ||
				RasterizerDiscard != b.RasterizerDiscard ||
				ColorMask != b.ColorMask ||
				BlendOperation != b.BlendOperation ||
				PolygonOffsetConstantMultiplier != b.PolygonOffsetConstantMultiplier ||
				PolygonOffsetGradientMultiplier != b.PolygonOffsetGradientMultiplier;
			for (uint32_t i=0; (i<MATERIAL_MAX_TEXTURES) && !different; ++i)
			{
				different |= (TextureLayer[i] != b.TextureLayer[i]);
			}
			return different;
		}

		//! Equality operator
		/** \param b Material to compare to.
		\return True if the materials are equal, else false. */
		inline bool operator==(const SMaterial& b) const
		{ return !(b!=*this); }

		bool isTransparent() const
		{
			return MaterialType==EMT_TRANSPARENT_ADD_COLOR ||
				MaterialType==EMT_TRANSPARENT_ALPHA_CHANNEL ||
				MaterialType==EMT_TRANSPARENT_VERTEX_ALPHA;
		}
	};

	//! global const identity Material
	IRRLICHT_API extern SMaterial IdentityMaterial;

} // end namespace video
} // end namespace irr

#endif
