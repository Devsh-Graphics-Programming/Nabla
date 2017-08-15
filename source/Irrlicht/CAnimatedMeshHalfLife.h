// Copyright (C) 2002-2012 Thomas Alten
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_ANIMATED_MESH_HALFLIFE_H_INCLUDED__
#define __C_ANIMATED_MESH_HALFLIFE_H_INCLUDED__

#include "IAnimatedMesh.h"
#include "ISceneManager.h"
#include "irrArray.h"
#include <string>
#include "IMeshLoader.h"
#include "SMesh.h"
#include "IReadFile.h"

namespace irr
{
namespace scene
{


	// STUDIO MODELS, Copyright (c) 1998, Valve LLC. All rights reserved.
	#define MAXSTUDIOTRIANGLES	20000	// TODO: tune this
	#define MAXSTUDIOVERTS		2048	// TODO: tune this
	#define MAXSTUDIOSEQUENCES	256		// total animation sequences
	#define MAXSTUDIOSKINS		100		// total textures
	#define MAXSTUDIOSRCBONES	512		// bones allowed at source movement
	#define MAXSTUDIOBONES		128		// total bones actually used
	#define MAXSTUDIOMODELS		32		// sub-models per model
	#define MAXSTUDIOBODYPARTS	32
	#define MAXSTUDIOGROUPS		4
	#define MAXSTUDIOANIMATIONS	512		// per sequence
	#define MAXSTUDIOMESHES		256
	#define MAXSTUDIOEVENTS		1024
	#define MAXSTUDIOPIVOTS		256
	#define MAXSTUDIOCONTROLLERS 8

	typedef float vec3_hl[3];	// x,y,z
	typedef float vec4_hl[4];	// x,y,z,w

// byte-align structures
#include "irrpack.h"

	struct SHalflifeHeader
	{
		int8_t id[4];
		int32_t version;

		int8_t name[64];
		int32_t length;

		vec3_hl eyeposition;	// ideal eye position
		vec3_hl min;			// ideal movement hull size
		vec3_hl max;

		vec3_hl bbmin;			// clipping bounding box
		vec3_hl bbmax;

		int32_t	flags;

		uint32_t	numbones;			// bones
		uint32_t	boneindex;

		uint32_t	numbonecontrollers;		// bone controllers
		uint32_t	bonecontrollerindex;

		uint32_t	numhitboxes;			// complex bounding boxes
		uint32_t	hitboxindex;

		uint32_t	numseq;				// animation sequences
		uint32_t	seqindex;

		uint32_t	numseqgroups;		// demand loaded sequences
		uint32_t	seqgroupindex;

		uint32_t	numtextures;		// raw textures
		uint32_t	textureindex;
		uint32_t	texturedataindex;

		uint32_t	numskinref;			// replaceable textures
		uint32_t	numskinfamilies;
		uint32_t	skinindex;

		uint32_t	numbodyparts;
		uint32_t	bodypartindex;

		uint32_t	numattachments;		// queryable attachable points
		uint32_t	attachmentindex;

		int32_t	soundtable;
		int32_t	soundindex;
		int32_t	soundgroups;
		int32_t	soundgroupindex;

		int32_t numtransitions;		// animation node to animation node transition graph
		int32_t	transitionindex;
	} PACK_STRUCT;

	// header for demand loaded sequence group data
	struct studioseqhdr_t
	{
		int32_t id;
		int32_t version;

		int8_t name[64];
		int32_t length;
	} PACK_STRUCT;

	// bones
	struct SHalflifeBone
	{
		int8_t name[32];	// bone name for symbolic links
		int32_t parent;		// parent bone
		int32_t flags;		// ??
		int32_t bonecontroller[6];	// bone controller index, -1 == none
		float value[6];	// default DoF values
		float scale[6];   // scale for delta DoF values
	} PACK_STRUCT;


	// bone controllers
	struct SHalflifeBoneController
	{
		int32_t bone;	// -1 == 0
		int32_t type;	// X, Y, Z, XR, YR, ZR, M
		float start;
		float end;
		int32_t rest;	// byte index value at rest
		int32_t index;	// 0-3 user set controller, 4 mouth
	} PACK_STRUCT;

	// intersection boxes
	struct SHalflifeBBox
	{
		int32_t bone;
		int32_t group;			// intersection group
		vec3_hl bbmin;		// bounding box
		vec3_hl bbmax;
	} PACK_STRUCT;

#ifndef ZONE_H
	// NOTE: this was a void*, but that crashes on 64bit.
	// I have found no mdl format desc, so not sure what it's meant to be, but int32_t at least works.
	typedef int32_t cache_user_t;
#endif

	// demand loaded sequence groups
	struct SHalflifeSequenceGroup
	{
		int8_t label[32];	// textual name
		int8_t name[64];	// file name
		cache_user_t cache;		// cache index pointer
		int32_t data;		// hack for group 0
	} PACK_STRUCT;

	// sequence descriptions
	struct SHalflifeSequence
	{
		int8_t label[32];	// sequence label

		float fps;		// frames per second
		int32_t flags;		// looping/non-looping flags

		int32_t activity;
		int32_t actweight;

		int32_t numevents;
		int32_t eventindex;

		int32_t numframes;	// number of frames per sequence

		uint32_t numpivots;	// number of foot pivots
		uint32_t pivotindex;

		int32_t motiontype;
		int32_t motionbone;
		vec3_hl linearmovement;
		int32_t automoveposindex;
		int32_t automoveangleindex;

		vec3_hl bbmin;		// per sequence bounding box
		vec3_hl bbmax;

		int32_t numblends;
		int32_t animindex;		// SHalflifeAnimOffset pointer relative to start of sequence group data
		// [blend][bone][X, Y, Z, XR, YR, ZR]

		int32_t blendtype[2];	// X, Y, Z, XR, YR, ZR
		float blendstart[2];	// starting value
		float blendend[2];	// ending value
		int32_t blendparent;

		int32_t seqgroup;		// sequence group for demand loading

		int32_t entrynode;		// transition node at entry
		int32_t exitnode;		// transition node at exit
		int32_t nodeflags;		// transition rules

		int32_t nextseq;		// auto advancing sequences
	} PACK_STRUCT;

	// events
	struct mstudioevent_t
	{
		int32_t frame;
		int32_t event;
		int32_t type;
		int8_t options[64];
	} PACK_STRUCT;


	// pivots
	struct mstudiopivot_t
	{
		vec3_hl org;	// pivot point
		int32_t start;
		int32_t end;
	} PACK_STRUCT;

	// attachment
	struct SHalflifeAttachment
	{
		int8_t name[32];
		int32_t type;
		int32_t bone;
		vec3_hl org;	// attachment point
		vec3_hl vectors[3];
	} PACK_STRUCT;

	struct SHalflifeAnimOffset
	{
		uint16_t	offset[6];
	} PACK_STRUCT;

	// animation frames
	union SHalflifeAnimationFrame
	{
		struct {
			uint8_t	valid;
			uint8_t	total;
		} PACK_STRUCT num;
		int16_t		value;
	} PACK_STRUCT;


	// body part index
	struct SHalflifeBody
	{
		int8_t name[64];
		uint32_t nummodels;
		uint32_t base;
		uint32_t modelindex; // index into models array
	} PACK_STRUCT;


	// skin info
	struct SHalflifeTexture
	{
		int8_t name[64];
		int32_t flags;
		int32_t width;
		int32_t height;
		int32_t index;
	} PACK_STRUCT;


	// skin families
	// short	index[skinfamilies][skinref]

	// studio models
	struct SHalflifeModel
	{
		int8_t name[64];
		int32_t type;

		float	boundingradius;

		uint32_t	nummesh;
		uint32_t	meshindex;

		uint32_t	numverts;		// number of unique vertices
		uint32_t	vertinfoindex;	// vertex bone info
		uint32_t	vertindex;		// vertex vec3_hl
		uint32_t	numnorms;		// number of unique surface normals
		uint32_t	norminfoindex;	// normal bone info
		uint32_t	normindex;		// normal vec3_hl

		uint32_t	numgroups;		// deformation groups
		uint32_t	groupindex;
	} PACK_STRUCT;


	// meshes
	struct SHalflifeMesh
	{
		uint32_t	numtris;
		uint32_t	triindex;
		uint32_t	skinref;
		uint32_t	numnorms;		// per mesh normals
		uint32_t	normindex;		// normal vec3_hl
	} PACK_STRUCT;

// Default alignment
#include "irrunpack.h"

	// lighting options
	#define STUDIO_NF_FLATSHADE		0x0001
	#define STUDIO_NF_CHROME		0x0002
	#define STUDIO_NF_FULLBRIGHT	0x0004

	// motion flags
	#define STUDIO_X		0x0001
	#define STUDIO_Y		0x0002
	#define STUDIO_Z		0x0004
	#define STUDIO_XR		0x0008
	#define STUDIO_YR		0x0010
	#define STUDIO_ZR		0x0020
	#define STUDIO_LX		0x0040
	#define STUDIO_LY		0x0080
	#define STUDIO_LZ		0x0100
	#define STUDIO_AX		0x0200
	#define STUDIO_AY		0x0400
	#define STUDIO_AZ		0x0800
	#define STUDIO_AXR		0x1000
	#define STUDIO_AYR		0x2000
	#define STUDIO_AZR		0x4000
	#define STUDIO_TYPES	0x7FFF
	#define STUDIO_RLOOP	0x8000	// controller that wraps shortest distance

	// sequence flags
	#define STUDIO_LOOPING	0x0001

	// bone flags
	#define STUDIO_HAS_NORMALS	0x0001
	#define STUDIO_HAS_VERTICES 0x0002
	#define STUDIO_HAS_BBOX		0x0004
	#define STUDIO_HAS_CHROME	0x0008	// if any of the textures have chrome on them

	#define RAD_TO_STUDIO		(32768.0/M_PI)
	#define STUDIO_TO_RAD		(M_PI/32768.0)

	/*!
		Textureatlas
		Combine Source Images with arbitrary size and bithdepth to an Image with 2^n size
		borders from the source images are copied around for allowing filtering ( bilinear, mipmap )
	*/
	struct STextureAtlas
	{
		STextureAtlas ()
		{
			release();
		}

		virtual ~STextureAtlas ()
		{
			release ();
		}

		void release ();
		void addSource ( const int8_t * name, video::IImage * image );
		void create ( uint32_t pixelborder, video::E_TEXTURE_CLAMP texmode );
		void getScale ( core::vector2df &scale );
		void getTranslation ( const int8_t * name, core::vector2di &pos );

		struct TextureAtlasEntry
		{
			io::path name;
			uint32_t width;
			uint32_t height;

			core::vector2di pos;

			video::IImage * image;

			bool operator < ( const TextureAtlasEntry & other )
			{
				return height > other.height;
			}
		};


		core::array < TextureAtlasEntry > atlas;
		video::IImage * Master;
	};


	//! Possible types of Animation Type
	enum E_ANIMATION_TYPE
	{
		//! No Animation
		EAMT_STILL,
		//! From Start to End, then Stop ( Limited Line )
		EAMT_WAYPOINT,
		//! Linear Cycling Animation	 ( Sawtooth )
		EAMT_LOOPING,
		//! Linear bobbing				 ( Triangle )
		EAMT_PINGPONG
	};

	//! Names for Animation Type
	const int8_t* const MeshAnimationTypeNames[] =
	{
		"still",
		"waypoint",
		"looping",
		"pingpong",
		0
	};


	//! Data for holding named Animation Info
	struct KeyFrameInterpolation
	{
		std::string Name;		// Name of the current Animation/Bone
		E_ANIMATION_TYPE AnimationType;	// Type of Animation ( looping, usw..)

		float CurrentFrame;		// Current Frame
		int32_t NextFrame;			// Frame which will be used next. For blending

		int32_t StartFrame;			// Absolute Frame where the current animation start
		int32_t Frames;				// Relative Frames how much Frames this animation have
		int32_t LoopingFrames;		// How much of Frames sould be looped
		int32_t EndFrame;			// Absolute Frame where the current animation ends End = start + frames - 1

		float FramesPerSecond;	// Speed in Frames/Seconds the animation is played
		float RelativeSpeed;		// Factor Original fps is modified

		uint32_t BeginTime;			// Animation started at this thime
		uint32_t EndTime;			// Animation end at this time
		uint32_t LastTime;			// Last Keyframe was done at this time

		KeyFrameInterpolation ( const int8_t * name = "", int32_t start = 0, int32_t frames = 0, int32_t loopingframes = 0,
								float fps = 0.f, float relativefps = 1.f  )
			: Name ( name ), AnimationType ( loopingframes ? EAMT_LOOPING : EAMT_WAYPOINT),
			CurrentFrame ( (float) start ), NextFrame ( start ), StartFrame ( start ),
			Frames ( frames ), LoopingFrames ( loopingframes ), EndFrame ( start + frames - 1 ),
			FramesPerSecond ( fps ), RelativeSpeed ( relativefps ),
			BeginTime ( 0 ), EndTime ( 0 ), LastTime ( 0 )
		{
		}

		// linear search
		bool operator == ( const KeyFrameInterpolation & other ) const
		{
			return equalsIgnoreCase<std::string>(Name,other.Name);
		}

	};


	//! a List holding named Animations
	typedef core::array < KeyFrameInterpolation > IAnimationList;

	//! a List holding named Skins
	typedef core::array < std::string > ISkinList;


	// Current Model per Body
	struct SubModel
	{
		std::string name;
		uint32_t startBuffer;
		uint32_t endBuffer;
		uint32_t state;
	};

	struct BodyPart
	{
		std::string name;
		uint32_t defaultModel;
		core::array < SubModel > model;
	};
	//! a List holding named Models and SubModels
	typedef core::array < BodyPart > IBodyList;

#ifndef NEW_MESHES
	class CAnimatedMeshHalfLife : public IAnimatedMesh
	{
	public:

		//! constructor
		CAnimatedMeshHalfLife();

		//! destructor
		virtual ~CAnimatedMeshHalfLife();

		//! loads a Halflife mdl file
		virtual bool loadModelFile( io::IReadFile* file, ISceneManager * smgr );

		//IAnimatedMesh
		virtual uint32_t getFrameCount() const;
		virtual IMesh* getMesh(int32_t frame, int32_t detailLevel);
		virtual const core::aabbox3d<float>& getBoundingBox() const;
		virtual E_ANIMATED_MESH_TYPE getMeshType() const;
		virtual void renderModel ( uint32_t param, video::IVideoDriver * driver, const core::matrix4 &absoluteTransformation);

		//! returns amount of mesh buffers.
		virtual uint32_t getMeshBufferCount() const;
		//! returns pointer to a mesh buffer
		virtual IMeshBuffer* getMeshBuffer(uint32_t nr) const;
		//! Returns pointer to a mesh buffer which fits a material
		virtual IMeshBuffer* getMeshBuffer( const video::SMaterial &material) const;

		virtual void setMaterialFlag(video::E_MATERIAL_FLAG flag, bool newvalue);

		//! set the hardware mapping hint, for driver
		virtual void setHardwareMappingHint(E_HARDWARE_MAPPING newMappingHint, E_BUFFER_TYPE buffer=EBT_VERTEX_AND_INDEX);

		//! flags the meshbuffer as changed, reloads hardware buffers
		virtual void setDirty(E_BUFFER_TYPE buffer=EBT_VERTEX_AND_INDEX);

		//! set user axis aligned bounding box
		virtual void setBoundingBox(const core::aabbox3df& box);

		//! Gets the default animation speed of the animated mesh.
		/** \return Amount of frames per second. If the amount is 0, it is a static, non animated mesh. */
		virtual float getAnimationSpeed() const
		{
			return FramesPerSecond;
		}

		//! Gets the frame count of the animated mesh.
		/** \param fps Frames per second to play the animation with. If the amount is 0, it is not animated.
		The actual speed is set in the scene node the mesh is instantiated in.*/
		virtual void setAnimationSpeed(float fps)
		{
			FramesPerSecond=fps;
		}

		//! Get the Animation List
		virtual IAnimationList* getAnimList () { return &AnimList; }

		//! Return the named Body List of this Animated Mesh
		virtual IBodyList *getBodyList() { return &BodyList; }

	private:

		// KeyFrame Animation List
		IAnimationList AnimList;
		// Sum of all sequences
		uint32_t FrameCount;

		// Named meshes of the Body
		IBodyList BodyList;

		//! return a Mesh per frame
		SMesh* MeshIPol;

		ISceneManager *SceneManager;

		SHalflifeHeader *Header;
		SHalflifeHeader *TextureHeader;
		bool OwnTexModel;						// do we have a modelT.mdl ?
		SHalflifeHeader *AnimationHeader[32];	// sequences named model01.mdl, model02.mdl

		void initData ();
		SHalflifeHeader * loadModel( io::IReadFile* file, const io::path &filename );
		bool postLoadModel( const io::path &filename );

		uint32_t SequenceIndex;	// sequence index
		float CurrentFrame;	// Current Frame
		float FramesPerSecond;

		#define MOUTH_CONTROLLER	4
		uint8_t BoneController[4 + 1 ]; // bone controllers + mouth position
		uint8_t Blending[2]; // animation blending

		vec4_hl BoneAdj;
		float SetController( int32_t controllerIndex, float value );

		uint32_t SkinGroupSelection; // skin group selection
		uint32_t SetSkin( uint32_t value );

		void initModel();
		void dumpModelInfo(uint32_t level) const;

		void ExtractBbox(int32_t sequence, core::aabbox3df &box) const;

		void setUpBones ();
		SHalflifeAnimOffset * getAnim( SHalflifeSequence *seq );
		void slerpBones( vec4_hl q1[], vec3_hl pos1[], vec4_hl q2[], vec3_hl pos2[], float s );
		void calcRotations ( vec3_hl *pos, vec4_hl *q, SHalflifeSequence *seq, SHalflifeAnimOffset *anim, float f );

		void calcBoneAdj();
_
#error "Fix QUATERNIONS FIRST!!!"
		void calcBoneQuaternion(const int32_t frame, const SHalflifeBone *bone, SHalflifeAnimOffset *anim, const uint32_t j, float& angle1, float& angle2) const;
		void calcBonePosition(const int32_t frame, float s, const SHalflifeBone *bone, SHalflifeAnimOffset *anim, float *pos ) const;

		void buildVertices ();

		io::path TextureBaseName;

#define HL_TEXTURE_ATLAS

#ifdef HL_TEXTURE_ATLAS
		STextureAtlas TextureAtlas;
		video::ITexture *TextureMaster;
#endif

	};


	//! Meshloader capable of loading HalfLife Model files
	class CHalflifeMDLMeshFileLoader : public IMeshLoader
	{
	public:

		//! Constructor
		CHalflifeMDLMeshFileLoader( scene::ISceneManager* smgr );

		//! returns true if the file maybe is able to be loaded by this class
		/** based on the file extension (e.g. ".bsp") */
		virtual bool isALoadableFileExtension(const io::path& filename) const;

		//! creates/loads an animated mesh from the file.
		/** \return Pointer to the created mesh. Returns 0 if loading failed.
		If you no longer need the mesh, you should call IAnimatedMesh::drop().
		See IReferenceCounted::drop() for more information.
		*/
		virtual IAnimatedMesh* createMesh(io::IReadFile* file);

	private:
		scene::ISceneManager* SceneManager;
	};
#endif // NEW_MESHES

} // end namespace scene
} // end namespace irr

#endif

