#ifdef __cplusplus

typedef uint32_t uint;
struct uvec2 { uint x, y; };
struct mat4 { float m00, m01, m02, m03, 
                    m10, m11, m12, m13, 
                    m20, m21, m22, m23, 
                    m30, m31, m32, m33; };

#endif // __cplusplus

struct RasterizerPushConstants {
	uvec2 imgSize;
	uint pointCount;
	uint totalThreads;
	mat4 mvp;
};

struct ShadingPushConstants {
	uvec2 imgSize;
};
