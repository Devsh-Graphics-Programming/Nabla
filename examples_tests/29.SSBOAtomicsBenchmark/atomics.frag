#if BINNING_METHOD==0

uint getAtomicOffset() {return 0u;}

#elif BINNING_METHOD==1

uint getAtomicOffset()
{
    uvec2 intPix = uvec2(gl_FragCoord.xy);
    uvec4 combined = intPix.xyxy&uvec4(1,7,14,8);
    combined = combined<<uvec4(0,1,3,4);
    uvec2 tmp = combined.xy|combined.zw;
    return tmp.x|tmp.y;
}

#elif BINNING_METHOD==2

uint getAtomicOffset()
{
    uvec2 intPix = uvec2(gl_FragCoord.xy);
    uvec3 combined = (intPix.xyx&uvec3(1,7,6))<<uvec3(0,1,3);
    return combined.x|combined.y|combined.z;
}

#elif BINNING_METHOD==3

#extension GL_ARB_shader_ballot: require
uint getAtomicOffset()
{
    uvec2 intPix = uvec2(gl_FragCoord.xy);
    uint sqrtSubGroupSize = findMSB(gl_SubGroupSizeARB);
    uvec2 cuMask = uvec2(kSqrtMaxConcurrentInvocations);
    cuMask -= uvec2(1u)<<uvec2(sqrtSubGroupSize/2+(sqrtSubGroupSize&0x1u),sqrtSubGroupSize/2);
    // everything above this line gets constant-folded by compiler
    intPix &= cuMask;
    intPix = (intPix^(intPix<<uvec2(2)))&uvec2(0x32u);
    intPix = (intPix^(intPix<<uvec2(1)))&uvec2(0x54u);
    return gl_SubGroupInvocationARB|intPix.x|(intPix.y<<1u);
}

#elif BINNING_METHOD==4

#extension GL_NV_shader_thread_group: require
uint getAtomicOffset()
{
    return gl_ThreadInWarpNV+(gl_SMIDNV<<5u);
}

#elif BINNING_METHOD==5

#extension GL_ARB_shader_ballot: require
uint getAtomicOffset()
{
    return gl_SubGroupInvocationARB;
}

#elif BINNING_METHOD==6

#extension GL_NV_shader_thread_group: require
uint getAtomicOffset()
{
    return gl_SMIDNV; // why is this faster than gl_ThreadInWarpNV???
}


#endif // BINNING_METHOD


uniform uint optimizerKillerOffset=256;


layout(std430, binding = 0) restrict buffer OutputAtomicData {
	uint packedHistogram[];
};


void main()
{
#if BINNING_METHOD==7
    packedHistogram[optimizerKillerOffset] = 1u;
#else
    atomicAdd(packedHistogram[optimizerKillerOffset+getAtomicOffset()],1u);
#endif // BINNING_METHOD
}

