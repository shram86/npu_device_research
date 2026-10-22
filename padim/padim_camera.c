#define _POSIX_C_SOURCE 200809L
/* PaDiM Experiment B: INT8 normalized ResNet input -> three dequantized CHW features.
 * Python owns statistics; optional native quadratic distances accelerate scoring.
 * No ONNX/Python dependency in this library. */
#include <vip_lite.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#define INPUT_ELEMENTS (3u*256u*256u)
#define FEATURE_ELEMENTS (64u*64u*64u + 128u*32u*32u + 256u*16u*16u)
#define REQUIRE(call) do { \
    vip_status_e status_ = (call); \
    if (status_ != VIP_SUCCESS) { \
        if (error && length) { \
            snprintf(error, length, "%s failed: VIP status %d", #call, (int)status_); \
        } \
        goto cleanup; \
    } \
} while (0)

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

/* Per-position quadratic distance. Precision rows are contiguous, allowing
 * vector loads instead of strided Python tensor contractions. No fast-math. */
int padim_mahalanobis(const float *delta,const float *precision,float *out,
                      size_t positions,size_t channels) {
    if(!delta || !precision || !out || positions!=4096 || channels!=100) return -1;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for(int p=0;p<4096;++p) {
        const float *d=delta+(size_t)p*100;
        const float *matrix=precision+(size_t)p*10000;
        float distance=0.0f;
        for(int i=0;i<100;++i) {
            const float *row=matrix+(size_t)i*100;
            float dot=0.0f;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
            float32x4_t sum=vdupq_n_f32(0.0f);
            for(int j=0;j<100;j+=4)
                sum=vaddq_f32(sum,vmulq_f32(vld1q_f32(row+j),vld1q_f32(d+j)));
            float lanes[4];vst1q_f32(lanes,sum);
            dot=(lanes[0]+lanes[1])+(lanes[2]+lanes[3]);
#else
            for(int j=0;j<100;++j) dot+=row[j]*d[j];
#endif
            distance+=d[i]*dot;
        }
        out[p]=distance;
    }
    return 0;
}

static inline float sample_bilinear(const uint8_t *src, int stride, int x0, int y0,
                                   int x1, int y1, float wx, float wy, int channel) {
    const uint8_t *top=src+(size_t)y0*(size_t)stride;
    const uint8_t *bottom=src+(size_t)y1*(size_t)stride;
    float a=top[(size_t)x0*3+(unsigned)channel], b=top[(size_t)x1*3+(unsigned)channel];
    float c=bottom[(size_t)x0*3+(unsigned)channel], d=bottom[(size_t)x1*3+(unsigned)channel];
    float upper=a+wx*(b-a), lower=c+wx*(d-c);
    return upper+wy*(lower-upper);
}

/* Fast sampling follows resize(shortest side=256), floor-centered crop=256.
 * Unlike PIL, it is point bilinear, without antialiasing or uint8 rounding.
 * Caller owns non-overlapping, sufficiently sized source/destination buffers. */
int padim_preprocess_neon_position(const uint8_t *src,int width,int height,int stride,
                          double position_x,double position_y,float *dst,size_t count) {
    if(!isfinite(position_x) || !isfinite(position_y) || position_x<0 || position_x>1 ||
       position_y<0 || position_y>1)return -1;
    if(!src || !dst || width<1 || height<1 || stride<1 ||
       (int64_t)stride < (int64_t)width*3 || count!=INPUT_ELEMENTS) return -1;
    int64_t resized_w=width<=height?256:(int64_t)256*width/height;
    int64_t resized_h=width<=height?(int64_t)256*height/width:256;
    if(resized_w>INT32_MAX || resized_h>INT32_MAX) return -1;
    int left=(int)((resized_w-256)*position_x),top=(int)((resized_h-256)*position_y);
    float scale_x=(float)width/(float)resized_w,scale_y=(float)height/(float)resized_h;
    const float means[3]={.485f,.456f,.406f};
    const float stds[3]={.229f,.224f,.225f};
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    const float32x4_t norm=vdupq_n_f32(1.f/255.f);
#endif
    /* Keep this small resize serial. Waking an OpenMP team immediately before
     * VIP readback can leave spinning workers competing with decode/capture.
     * NEON still accelerates normalization; only scoring uses OpenMP. */
    for(int oy=0;oy<256;oy++) {
        float sy=(oy+top+.5f)*scale_y-.5f;
        if(sy<0)sy=0;
        if(sy>height-1)sy=(float)(height-1);
        int y0=(int)sy,y1=y0+1<height?y0+1:y0;
        float wy=sy-y0;
        for(int ox=0;ox<256;ox+=4) {
            float colors[3][4];
            for(int lane=0;lane<4;lane++) {
                float sx=(ox+lane+left+.5f)*scale_x-.5f;
                if(sx<0)sx=0;
                if(sx>width-1)sx=(float)(width-1);
                int x0=(int)sx,x1=x0+1<width?x0+1:x0;
                float wx=sx-x0;
                for(int channel=0;channel<3;channel++)
                    colors[channel][lane]=sample_bilinear(src,stride,x0,y0,x1,y1,wx,wy,2-channel);
            }
            for(int channel=0;channel<3;channel++) {
                size_t offset=(size_t)channel*256*256+(size_t)oy*256+(unsigned)ox;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                float32x4_t value=vmulq_f32(vld1q_f32(colors[channel]),norm);
                value=vsubq_f32(value,vdupq_n_f32(means[channel]));
                value=vmulq_f32(value,vdupq_n_f32(1.f/stds[channel]));
                vst1q_f32(dst+offset,value);
#else
                for(int lane=0;lane<4;lane++)
                    dst[offset+(unsigned)lane]=(colors[channel][lane]*(1.f/255.f)-means[channel])/stds[channel];
#endif
            }
        }
    }
    return 0;
}

/* Preserve the existing centered ABI for standalone callers. */
int padim_preprocess_neon(const uint8_t *src,int width,int height,int stride,
                          float *dst,size_t count) {
    return padim_preprocess_neon_position(src,width,height,stride,.5,.5,dst,count);
}

typedef struct {
    vip_network network;
    vip_buffer input, outputs[3];
    vip_buffer_create_params_t ip, op[3];
    unsigned layer_port[3];
    int initialized, prepared, timings_valid;
    double timings[3];
    int selected_indices[100],selection_ready;
} PadimContext;
static double now_ms(void) {
    struct timespec t; if(clock_gettime(CLOCK_MONOTONIC,&t))return NAN;
    return (double)t.tv_sec*1000+(double)t.tv_nsec/1e6;
}
static int fail(char *error,size_t length,const char *message) {
    if(error&&length)snprintf(error,length,"%s",message);
    return -1;
}
static size_t elements(const vip_buffer_create_params_t *p) {
    size_t count = 1;
    if (!p->num_of_dims || p->num_of_dims > sizeof(p->sizes)/sizeof(p->sizes[0])) return 0;
    for (unsigned i = 0; i < p->num_of_dims; ++i) {
        if (!p->sizes[i] || count > SIZE_MAX / p->sizes[i]) return 0;
        count *= p->sizes[i];
    }
    return count;
}

static float half_to_float(uint16_t bits) {
    unsigned exponent = (bits >> 10) & 31u, mantissa = bits & 1023u;
    float value = exponent == 31u ? (mantissa ? NAN : INFINITY) :
                  exponent ? ldexpf((float)(1024u + mantissa), (int)exponent - 25) :
                             ldexpf((float)mantissa, -24);
    return bits & 0x8000u ? -value : value;
}

static int element_bytes(const vip_buffer_create_params_t *p) {
    switch (p->data_format) {
        case VIP_BUFFER_FORMAT_FP32: return 4;
        case VIP_BUFFER_FORMAT_FP16:
        case VIP_BUFFER_FORMAT_INT16: return 2;
        case VIP_BUFFER_FORMAT_INT8:
        case VIP_BUFFER_FORMAT_UINT8: return 1;
        default: return 0;
    }
}

static int quantization(const vip_buffer_create_params_t *p, float *scale, int32_t *zero) {
    *zero = 0;
    if (p->data_format == VIP_BUFFER_FORMAT_FP32 || p->data_format == VIP_BUFFER_FORMAT_FP16) {
        *scale = 1;
        return p->quant_format == VIP_BUFFER_QUANTIZE_NONE ? 0 : -1;
    }
    if (!element_bytes(p)) return -1;
    if (p->quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT)
        *scale = ldexpf(1.0f, -(int)(int8_t)p->quant_data.dfp.fixed_point_pos);
    else if (p->quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM) {
        *scale = p->quant_data.affine.scale;
        *zero = p->quant_data.affine.zeroPoint;
    } else return -1;
    return isfinite(*scale) && *scale > 0 ? 0 : -1;
}

static int fill_params(vip_network net, int output, unsigned index,
                       vip_buffer_create_params_t *p) {
    vip_uint32_t dims = 0;
    vip_enum format = 0, quant = 0;
    memset(p, 0, sizeof(*p));
#define QUERY(prop, ptr) ((output ? vip_query_output(net,index,prop,ptr) : \
                                  vip_query_input(net,index,prop,ptr)) == VIP_SUCCESS)
    if (!QUERY(VIP_BUFFER_PROP_NUM_OF_DIMENSION, &dims) || !dims ||
        dims > sizeof(p->sizes)/sizeof(p->sizes[0])) return -1;
    if (!QUERY(VIP_BUFFER_PROP_SIZES_OF_DIMENSION, p->sizes) ||
        !QUERY(VIP_BUFFER_PROP_DATA_FORMAT, &format) ||
        !QUERY(VIP_BUFFER_PROP_QUANT_FORMAT, &quant)) return -1;
    p->num_of_dims = dims;
    p->data_format = format;
    p->quant_format = quant;
    p->memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT;
    if (quant == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
        vip_uint8_t fp = 0;
        if (!QUERY(VIP_BUFFER_PROP_FIXED_POINT_POS, &fp)) return -1;
        p->quant_data.dfp.fixed_point_pos = fp;
    } else if (quant == VIP_BUFFER_QUANTIZE_TF_ASYMM) {
        vip_float_t scale = 0;
        vip_uint32_t zero = 0;
        if (!QUERY(VIP_BUFFER_PROP_TF_SCALE, &scale) ||
            !QUERY(VIP_BUFFER_PROP_TF_ZERO_POINT, &zero)) return -1;
        p->quant_data.affine.scale = scale;
        p->quant_data.affine.zeroPoint = (vip_int32_t)zero;
    }
#undef QUERY
    return 0;
}

void padim_close(void *handle) {
    PadimContext *c=handle; if(!c)return;
    for(unsigned i=0;i<3;i++)if(c->outputs[i])vip_destroy_buffer(c->outputs[i]);
    if(c->input)vip_destroy_buffer(c->input);
    if(c->prepared)vip_finish_network(c->network);
    if(c->network)vip_destroy_network(c->network);
    if(c->initialized)vip_destroy();
    free(c);
}
static int is_shape(const vip_buffer_create_params_t *p,unsigned w,unsigned h,unsigned channels) {
    return (p->num_of_dims==3 || (p->num_of_dims==4 && p->sizes[3]==1)) &&
           p->sizes[0]==w && p->sizes[1]==h && p->sizes[2]==channels;
}
void *padim_open(const char *filename,char *error,size_t length) {
    PadimContext *c=calloc(1,sizeof(*c));
    vip_uint32_t ni=0,no=0; float scale; int32_t zero; unsigned seen=0;
    if(!c){fail(error,length,"Out of memory");return NULL;}
    if(!filename){fail(error,length,"Missing NB filename");goto cleanup;}
    REQUIRE(vip_init());c->initialized=1;
    REQUIRE(vip_create_network(filename,0,VIP_CREATE_NETWORK_FROM_FILE,&c->network));
    REQUIRE(vip_query_network(c->network,VIP_NETWORK_PROP_INPUT_COUNT,&ni));
    REQUIRE(vip_query_network(c->network,VIP_NETWORK_PROP_OUTPUT_COUNT,&no));
    if(ni!=1 || no!=3){fail(error,length,"Expected one input and three ResNet feature outputs");goto cleanup;}
    if(fill_params(c->network,0,0,&c->ip) || !is_shape(&c->ip,256,256,3) ||
       c->ip.data_format!=VIP_BUFFER_FORMAT_INT8 || c->ip.quant_format!=VIP_BUFFER_QUANTIZE_TF_ASYMM ||
       quantization(&c->ip,&scale,&zero)) {
        fail(error,length,"Expected signed INT8 affine NCHW [1,3,256,256] input");goto cleanup;
    }
    for(unsigned port=0;port<3;port++) {
        vip_buffer_create_params_t *p=&c->op[port];
        if(fill_params(c->network,1,port,p) || quantization(p,&scale,&zero)) {
            fail(error,length,"Unsupported feature storage/quantization");goto cleanup;
        }
        unsigned layer=3;
        if(is_shape(p,64,64,64))layer=0;
        if(is_shape(p,32,32,128))layer=1;
        if(is_shape(p,16,16,256))layer=2;
        if(layer==3 || (seen&(1u<<layer))) {
            fail(error,length,"Expected unique NCHW layer1/2/3 feature shapes; no tensor batching");goto cleanup;
        }
        c->layer_port[layer]=port;seen|=1u<<layer;
        fprintf(stderr,"PaDiM layer%u <- output %u: %ux%ux%u type=%d scale=%.9g zero=%d\n",
                layer+1,port,p->sizes[2],p->sizes[1],p->sizes[0],(int)p->data_format,(double)scale,(int)zero);
    }
    REQUIRE(vip_prepare_network(c->network));c->prepared=1;
    REQUIRE(vip_create_buffer(&c->ip,sizeof(c->ip),&c->input));
    if(vip_get_buffer_size(c->input)!=INPUT_ELEMENTS){fail(error,length,"Padded input unsupported");goto cleanup;}
    REQUIRE(vip_set_input(c->network,0,c->input));
    for(unsigned i=0;i<3;i++) {
        REQUIRE(vip_create_buffer(&c->op[i],sizeof(c->op[i]),&c->outputs[i]));
        size_t bytes=elements(&c->op[i])*(size_t)element_bytes(&c->op[i]);
        if(!bytes || vip_get_buffer_size(c->outputs[i])!=bytes){fail(error,length,"Padded output unsupported");goto cleanup;}
        REQUIRE(vip_set_output(c->network,i,c->outputs[i]));
    }
    return c;
cleanup:
    padim_close(c);return NULL;
}
/* index -1=input, 0..2=semantic layer1..3. Returns floating storage flag, scale, zero. */
int padim_io_info(void *handle,int index,double *info,size_t count) {
    PadimContext *c=handle;
    if(!c || !info || count!=3 || index < -1 || index > 2)return -1;
    vip_buffer_create_params_t *p=index<0?&c->ip:&c->op[c->layer_port[index]];
    float scale;int32_t zero;
    if(quantization(p,&scale,&zero))return -1;
    info[0]=p->data_format==VIP_BUFFER_FORMAT_FP32 || p->data_format==VIP_BUFFER_FORMAT_FP16;
    info[1]=scale;info[2]=zero;return 0;
}
int padim_get_timings(void *handle,double *values,size_t count) {
    PadimContext *c=handle;
    if(!c || !values || count!=3 || !c->timings_valid)return -1;
    memcpy(values,c->timings,sizeof(c->timings));return 0;
}
/* Save NPZ channel order once. Reject duplicate/out-of-range selections. */
int padim_set_channels(void *handle,const int32_t *indices,size_t count) {
    PadimContext *c=handle;unsigned char seen[448]={0};
    if(!c || !indices || count!=100)return -1;
    for(size_t i=0;i<count;i++) {
        if(indices[i]<0 || indices[i]>=448 || seen[indices[i]])return -1;
        seen[indices[i]]=1;
    }
    memcpy(c->selected_indices,indices,sizeof(c->selected_indices));c->selection_ready=1;
    return 0;
}
static float decode_feature(const void *src,size_t i,const vip_buffer_create_params_t *p,
                            float scale,int32_t zero) {
    if(p->data_format==VIP_BUFFER_FORMAT_FP32)return ((const float *)src)[i];
    if(p->data_format==VIP_BUFFER_FORMAT_FP16)return half_to_float(((const uint16_t *)src)[i]);
    int32_t q=p->data_format==VIP_BUFFER_FORMAT_INT8?((const int8_t *)src)[i]:
              p->data_format==VIP_BUFFER_FORMAT_UINT8?((const uint8_t *)src)[i]:((const int16_t *)src)[i];
    return ((float)q-(float)zero)*scale;
}
static int read_selected(const void *src,const vip_buffer_create_params_t *p,unsigned layer,
                         const int *indices,float *dst,float scale,int32_t zero) {
    static const int offsets[3]={0,64,192},channels[3]={64,128,256};
    int width=(int)p->sizes[0],hw=width*width;
    float channel[4096],lookup[256];
    int use_lookup=p->data_format==VIP_BUFFER_FORMAT_INT8 || p->data_format==VIP_BUFFER_FORMAT_UINT8;
    if(use_lookup)for(int q=0;q<256;q++) {
        int value=p->data_format==VIP_BUFFER_FORMAT_INT8?q-128:q;
        lookup[q]=((float)value-(float)zero)*scale;
    }
    for(int selected=0;selected<100;selected++) {
        int ch=indices[selected]-offsets[layer];
        if(ch<0 || ch>=channels[layer])continue;
        for(int i=0;i<hw;i++) {
            size_t at=(size_t)ch*(size_t)hw+(unsigned)i;
            float value;
            if(use_lookup) {
                int q=p->data_format==VIP_BUFFER_FORMAT_INT8?(int)((const int8_t *)src)[at]+128:
                                                          ((const uint8_t *)src)[at];
                value=lookup[q];
            } else value=decode_feature(src,at,p,scale,zero);
            if(!isfinite(value))return -1;
            channel[i]=value;
        }
        float *out=dst+(size_t)selected*4096;
        if(width==64){memcpy(out,channel,4096*sizeof(float));continue;}
        /* PyTorch interpolate: half-pixel centers, clamped edges, align_corners=False. */
        for(int y=0;y<64;y++) {
            float sy=(y+.5f)*(float)width/64.f-.5f;
            if(sy<0)sy=0;
            if(sy>width-1)sy=(float)(width-1);
            int y0=(int)sy,y1=y0+1<width?y0+1:y0;float wy=sy-y0;
            for(int x=0;x<64;x++) {
                float sx=(x+.5f)*(float)width/64.f-.5f;
                if(sx<0)sx=0;
                if(sx>width-1)sx=(float)(width-1);
                int x0=(int)sx,x1=x0+1<width?x0+1:x0;float wx=sx-x0;
                float top=channel[y0*width+x0]*(1-wx)+channel[y0*width+x1]*wx;
                float bottom=channel[y1*width+x0]*(1-wx)+channel[y1*width+x1]*wx;
                out[y*64+x]=top*(1-wy)+bottom*wy;
                if(!isfinite(out[y*64+x]))return -1;
            }
        }
    }
    return 0;
}
static int run_features(void *handle,const int8_t *image,size_t input_count,float *features,size_t feature_count,
              char *error,size_t length,int selected) {
    PadimContext *c=handle;void *input=NULL,*output=NULL;unsigned mapped_port=0;
    double start=now_ms(),ready=0,done=0;int result=-1;
    if(!c || !image || !features || input_count!=INPUT_ELEMENTS || feature_count!=(selected?409600u:FEATURE_ELEMENTS) || (selected && !c->selection_ready))
        return fail(error,length,"Invalid PaDiM inference buffer size/context");
    c->timings_valid=0;
    input=vip_map_buffer(c->input);
    if(!input)return fail(error,length,"Cannot map input");
    memcpy(input,image,INPUT_ELEMENTS);
    vip_unmap_buffer(c->input);input=NULL;
    REQUIRE(vip_flush_buffer(c->input,VIP_BUFFER_OPER_TYPE_FLUSH));
    ready=now_ms();
    REQUIRE(vip_run_network(c->network));
    done=now_ms();
    size_t offset=0;
    for(unsigned layer=0;layer<3;layer++) {
        unsigned port=c->layer_port[layer];
        mapped_port=port;
        vip_buffer_create_params_t *p=&c->op[port];
        float scale;int32_t zero;
        quantization(p,&scale,&zero);
        REQUIRE(vip_flush_buffer(c->outputs[port],VIP_BUFFER_OPER_TYPE_INVALIDATE));
        output=vip_map_buffer(c->outputs[port]);
        if(!output){
            fail(error,length,"Cannot map output");
            goto cleanup;
        }
        size_t count=elements(p);
        if(selected) {
            if(read_selected(output,p,layer,c->selected_indices,features,scale,zero)) {
                fail(error,length,"Nonfinite selected feature output");goto cleanup;
            }
        } else for(size_t i=0;i<count;i++) {
            float value;
            if(p->data_format==VIP_BUFFER_FORMAT_FP32)value=((float *)output)[i];
            else if(p->data_format==VIP_BUFFER_FORMAT_FP16)value=half_to_float(((uint16_t *)output)[i]);
            else {
                int32_t q=p->data_format==VIP_BUFFER_FORMAT_INT8?((int8_t *)output)[i]:
                          p->data_format==VIP_BUFFER_FORMAT_UINT8?((uint8_t *)output)[i]:((int16_t *)output)[i];
                value=(float)(((double)q-zero)*scale);
            }
            if(!isfinite(value)){
                fail(error,length,"Nonfinite feature output");
                goto cleanup;
            }
            features[offset+i]=value;
        }
        vip_unmap_buffer(c->outputs[port]);output=NULL;offset+=count;
    }
    result=0;
cleanup:
    if(input)vip_unmap_buffer(c->input);
    if(output)vip_unmap_buffer(c->outputs[mapped_port]);
    if(!result) {
        c->timings[0]=ready-start;c->timings[1]=done-ready;c->timings[2]=now_ms()-done;
        c->timings_valid=isfinite(c->timings[0])&&isfinite(c->timings[1])&&isfinite(c->timings[2]);
    }
    return result;
}

int padim_run(void *handle,const int8_t *image,size_t input_count,float *features,size_t count,
              char *error,size_t length) {
    return run_features(handle,image,input_count,features,count,error,length,0);
}
int padim_run_selected(void *handle,const int8_t *image,size_t input_count,float *features,size_t count,
                       char *error,size_t length) {
    return run_features(handle,image,input_count,features,count,error,length,1);
}

/* A second graph shares the feature context's initialized VIP runtime.
 * Close it before padim_close(owner); it must never call vip_destroy itself. */
typedef struct {
    vip_network network;
    vip_buffer input,output;
    int prepared;
} ScoreContext;
void padim_score_close(void *handle) {
    ScoreContext *c=handle;if(!c)return;
    if(c->input)vip_destroy_buffer(c->input);
    if(c->output)vip_destroy_buffer(c->output);
    if(c->prepared)vip_finish_network(c->network);
    if(c->network)vip_destroy_network(c->network);
    free(c);
}
void *padim_score_open(void *owner,const char *filename,char *error,size_t length) {
    ScoreContext *c=NULL;vip_buffer_create_params_t ip,op;vip_uint32_t ni=0,no=0;
    if(!owner || !((PadimContext *)owner)->initialized || !filename) {
        fail(error,length,"Score graph requires a live feature context and NB path");return NULL;
    }
    c=calloc(1,sizeof(*c));if(!c){fail(error,length,"Out of memory");return NULL;}
    REQUIRE(vip_create_network(filename,0,VIP_CREATE_NETWORK_FROM_FILE,&c->network));
    REQUIRE(vip_query_network(c->network,VIP_NETWORK_PROP_INPUT_COUNT,&ni));
    REQUIRE(vip_query_network(c->network,VIP_NETWORK_PROP_OUTPUT_COUNT,&no));
    if(ni!=1 || no!=1 || fill_params(c->network,0,0,&ip) || fill_params(c->network,1,0,&op) ||
       !is_shape(&ip,64,64,100) || !is_shape(&op,64,64,1) ||
       ip.data_format!=VIP_BUFFER_FORMAT_FP16 || op.data_format!=VIP_BUFFER_FORMAT_FP16 ||
       ip.quant_format!=VIP_BUFFER_QUANTIZE_NONE || op.quant_format!=VIP_BUFFER_QUANTIZE_NONE) {
        fail(error,length,"Score NB must have FP16 NCHW [1,100,64,64] -> [1,1,64,64]");goto cleanup;
    }
    REQUIRE(vip_prepare_network(c->network));c->prepared=1;
    REQUIRE(vip_create_buffer(&ip,sizeof(ip),&c->input));
    REQUIRE(vip_create_buffer(&op,sizeof(op),&c->output));
    if(vip_get_buffer_size(c->input)!=409600u*2 || vip_get_buffer_size(c->output)!=4096u*2) {
        fail(error,length,"Padded score graph buffers unsupported");goto cleanup;
    }
    REQUIRE(vip_set_input(c->network,0,c->input));
    REQUIRE(vip_set_output(c->network,0,c->output));
    return c;
cleanup:
    padim_score_close(c);return NULL;
}
int padim_score_run(void *handle,const uint16_t *features,size_t count,uint16_t *distance,
                    size_t output_count,double *timings,char *error,size_t length) {
    ScoreContext *c=handle;void *mapped=NULL;double start=now_ms(),ready,done;
    if(!c || !features || !distance || !timings || count!=409600 || output_count!=4096)
        return fail(error,length,"Invalid score buffers");
    mapped=vip_map_buffer(c->input);
    if(!mapped)return fail(error,length,"Cannot map score input");
    memcpy(mapped,features,count*2);vip_unmap_buffer(c->input);mapped=NULL;
    REQUIRE(vip_flush_buffer(c->input,VIP_BUFFER_OPER_TYPE_FLUSH));
    ready=now_ms();REQUIRE(vip_run_network(c->network));done=now_ms();
    REQUIRE(vip_flush_buffer(c->output,VIP_BUFFER_OPER_TYPE_INVALIDATE));
    mapped=vip_map_buffer(c->output);
    if(!mapped)return fail(error,length,"Cannot map score output");
    memcpy(distance,mapped,output_count*2);vip_unmap_buffer(c->output);mapped=NULL;
    timings[0]=ready-start;timings[1]=done-ready;timings[2]=now_ms()-done;
    return 0;
cleanup:
    return -1;
}
