#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <signal.h>
#include <glob.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#include <vip_lite.h>

/* Discover by sensor name because thermal-zone numbers can vary by kernel. */
static void npu_temperature(char *text, size_t size)
{
    static char path[512];
    static int searched;
    if (!searched) {
        glob_t zones = {0};
        searched = 1;
        if (glob("/sys/class/thermal/thermal_zone*/type", 0, NULL, &zones) == 0) {
            for (size_t i = 0; i < zones.gl_pathc; ++i) {
                char type[128];
                FILE *f = fopen(zones.gl_pathv[i], "r");
                if (!f) continue;
                int ok = fscanf(f, "%127s", type) == 1;
                fclose(f);
                if (ok && strcmp(type, "npu_thermal_zone") == 0) {
                    snprintf(path, sizeof(path), "%s", zones.gl_pathv[i]);
                    char *name = strrchr(path, '/');
                    if (name) strcpy(name + 1, "temp");
                    break;
                }
            }
        }
        globfree(&zones);
    }
    snprintf(text, size, "N/A");
    FILE *f = path[0] ? fopen(path, "r") : NULL;
    if (f) {
        long millidegrees;
        if (fscanf(f, "%ld", &millidegrees) == 1)
            snprintf(text, size, "%.1f C", millidegrees / 1000.0);
        fclose(f);
    }
}


#define NBG_FILE "./network_binary.nb"
#define CAMERA_DEVICE "/dev/video0"
#define CAM_W 640
#define CAM_H 480
#define NET_W 256
#define NET_H 256
#define NUM_V4L2_BUFFERS 4

struct camera_buffer {
    void *start;
    size_t length;
};

/* fd 3 carries packed YUYV only: ./pwcnet_camera 3>&1 1>&2 | ffmpeg ... */
static volatile sig_atomic_t stopping = 0;
static void stop_capture(int signo)
{
    int saved_errno = errno;
    if (!stopping) {
        /* write is async-signal-safe; stdio is not. Keep video fd 3 clean. */
        static const char interrupt[] = "\nCtrl+C received: stopping BlazePose...\n";
        static const char terminate[] = "\nSIGTERM received: stopping BlazePose...\n";
        ssize_t written;
        if (signo == SIGINT)
            written = write(STDERR_FILENO, interrupt, sizeof(interrupt) - 1);
        else
            written = write(STDERR_FILENO, terminate, sizeof(terminate) - 1);
        (void)written;
    }
    stopping = 1;
    errno = saved_errno;
}

static int stream_frame(const uint8_t *src, size_t stride)
{
    if (stride != CAM_W * 2) { errno=EINVAL; return -1; }
    size_t remaining=(size_t)CAM_W*CAM_H*2;
    while (remaining && !stopping) {
        ssize_t n=write(3,src,remaining);
        if(n<0 && errno==EINTR) continue;
        if(n<0 && errno==EPIPE) { stopping=1; return 0; }
        if(n<=0) { perror("write video fd 3"); return -1; }
        src+=n;remaining-=(size_t)n;
    }
    return 0;
}

static uint8_t clip_u8(int x)
{
    if (x < 0)   return 0;
    if (x > 255) return 255;
    return (uint8_t)x;
}

#define CHECK(call) do {                                      \
    vip_status_e _s = (call);                                 \
    if (_s != VIP_SUCCESS) {                                  \
        fprintf(stderr, "%s failed: status=%d\n",             \
                #call, (int)_s);                              \
        goto cleanup;                                         \
    }                                                         \
} while (0)

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static void print_tensor(vip_network net, int output, uint32_t index)
{
    vip_uint32_t dims = 0;
    vip_uint32_t sizes[6] = {0};
    vip_enum format = 0;
    vip_enum quant = 0;
    vip_uint8_t fp = 0;
    vip_float_t scale = 0;
    vip_uint32_t zp = 0;
    char name[64] = {0};

    if (output) {
        vip_query_output(net, index, VIP_BUFFER_PROP_NUM_OF_DIMENSION, &dims);
        vip_query_output(net, index, VIP_BUFFER_PROP_SIZES_OF_DIMENSION, sizes);
        vip_query_output(net, index, VIP_BUFFER_PROP_DATA_FORMAT, &format);
        vip_query_output(net, index, VIP_BUFFER_PROP_QUANT_FORMAT, &quant);
        vip_query_output(net, index, VIP_BUFFER_PROP_NAME, name);

        if (quant == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT)
            vip_query_output(net, index, VIP_BUFFER_PROP_FIXED_POINT_POS, &fp);
        else if (quant == VIP_BUFFER_QUANTIZE_TF_ASYMM) {
            vip_query_output(net, index, VIP_BUFFER_PROP_TF_SCALE, &scale);
            vip_query_output(net, index, VIP_BUFFER_PROP_TF_ZERO_POINT, &zp);
        }
    } else {
        vip_query_input(net, index, VIP_BUFFER_PROP_NUM_OF_DIMENSION, &dims);
        vip_query_input(net, index, VIP_BUFFER_PROP_SIZES_OF_DIMENSION, sizes);
        vip_query_input(net, index, VIP_BUFFER_PROP_DATA_FORMAT, &format);
        vip_query_input(net, index, VIP_BUFFER_PROP_QUANT_FORMAT, &quant);
        vip_query_input(net, index, VIP_BUFFER_PROP_NAME, name);

        if (quant == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT)
            vip_query_input(net, index, VIP_BUFFER_PROP_FIXED_POINT_POS, &fp);
        else if (quant == VIP_BUFFER_QUANTIZE_TF_ASYMM) {
            vip_query_input(net, index, VIP_BUFFER_PROP_TF_SCALE, &scale);
            vip_query_input(net, index, VIP_BUFFER_PROP_TF_ZERO_POINT, &zp);
        }
    }

    fprintf(stderr, "%s %u: name=%s format=%d quant=%d dims=",
           output ? "output" : "input",
           index, name, (int)format, (int)quant);

    for (uint32_t i = 0; i < dims; ++i)
        fprintf(stderr, "%u%s", sizes[i], i + 1 == dims ? "" : "x");

    if (quant == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT)
        fprintf(stderr, " DFP=%d", (int)(int8_t)fp);
    else if (quant == VIP_BUFFER_QUANTIZE_TF_ASYMM)
        fprintf(stderr, " scale=%.9g zp=%d", scale, (int32_t)zp);

    fprintf(stderr, "\n");
}

static int fill_params(vip_network net, int output, uint32_t index,
                       vip_buffer_create_params_t *p)
{
    vip_uint32_t dims = 0;
    vip_enum format = 0;
    vip_enum quant = 0;

    memset(p, 0, sizeof(*p));

    if (output) {
        if (vip_query_output(net, index,
                VIP_BUFFER_PROP_NUM_OF_DIMENSION, &dims) != VIP_SUCCESS)
            return -1;

        if (vip_query_output(net, index,
                VIP_BUFFER_PROP_SIZES_OF_DIMENSION, p->sizes) != VIP_SUCCESS)
            return -1;

        if (vip_query_output(net, index,
                VIP_BUFFER_PROP_DATA_FORMAT, &format) != VIP_SUCCESS)
            return -1;

        if (vip_query_output(net, index,
                VIP_BUFFER_PROP_QUANT_FORMAT, &quant) != VIP_SUCCESS)
            return -1;
    } else {
        if (vip_query_input(net, index,
                VIP_BUFFER_PROP_NUM_OF_DIMENSION, &dims) != VIP_SUCCESS)
            return -1;

        if (vip_query_input(net, index,
                VIP_BUFFER_PROP_SIZES_OF_DIMENSION, p->sizes) != VIP_SUCCESS)
            return -1;

        if (vip_query_input(net, index,
                VIP_BUFFER_PROP_DATA_FORMAT, &format) != VIP_SUCCESS)
            return -1;

        if (vip_query_input(net, index,
                VIP_BUFFER_PROP_QUANT_FORMAT, &quant) != VIP_SUCCESS)
            return -1;
    }

    p->num_of_dims = dims;
    p->data_format = format;
    p->quant_format = quant;
    p->memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT;

    if (quant == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
        vip_uint8_t fp = 0;

        if (output)
            vip_query_output(net, index,
                VIP_BUFFER_PROP_FIXED_POINT_POS, &fp);
        else
            vip_query_input(net, index,
                VIP_BUFFER_PROP_FIXED_POINT_POS, &fp);

        p->quant_data.dfp.fixed_point_pos = fp;
    }
    else if (quant == VIP_BUFFER_QUANTIZE_TF_ASYMM) {
        vip_float_t scale = 0;
        vip_uint32_t zp = 0;

        if (output) {
            vip_query_output(net, index,
                VIP_BUFFER_PROP_TF_SCALE, &scale);
            vip_query_output(net, index,
                VIP_BUFFER_PROP_TF_ZERO_POINT, &zp);
        } else {
            vip_query_input(net, index,
                VIP_BUFFER_PROP_TF_SCALE, &scale);
            vip_query_input(net, index,
                VIP_BUFFER_PROP_TF_ZERO_POINT, &zp);
        }

        p->quant_data.affine.scale = scale;
        p->quant_data.affine.zeroPoint = (vip_int32_t)zp;
    }

    return 0;
}

static int camera_open(
    const char *device,
    struct camera_buffer buffers[NUM_V4L2_BUFFERS], size_t *stride)
{
    int fd = open(device, O_RDWR);
    if (fd < 0) {
        perror("open camera");
        return -1;
    }

    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));

    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = CAM_W;
    fmt.fmt.pix.height = CAM_H;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
        perror("VIDIOC_S_FMT");
        close(fd);
        return -1;
    }

    fprintf(stderr,
        "camera negotiated: %ux%u fourcc=%c%c%c%c "
        "bytesperline=%u sizeimage=%u\n",
        fmt.fmt.pix.width,
        fmt.fmt.pix.height,
        fmt.fmt.pix.pixelformat & 0xff,
        (fmt.fmt.pix.pixelformat >> 8) & 0xff,
        (fmt.fmt.pix.pixelformat >> 16) & 0xff,
        (fmt.fmt.pix.pixelformat >> 24) & 0xff,
        fmt.fmt.pix.bytesperline,
        fmt.fmt.pix.sizeimage);

    if (fmt.fmt.pix.width != CAM_W || fmt.fmt.pix.height != CAM_H ||
        fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV ||
        fmt.fmt.pix.bytesperline < CAM_W * 2) {
        fprintf(stderr, "Camera must provide 640x480 YUYV with valid row stride\n");
        close(fd); return -1;
    }
    *stride = fmt.fmt.pix.bytesperline;

    struct v4l2_streamparm parm;
    memset(&parm, 0, sizeof(parm));

    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = 30;

    if (ioctl(fd, VIDIOC_S_PARM, &parm) < 0) {
        perror("VIDIOC_S_PARM");
        close(fd);
        return -1;
    }

    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));

    req.count = NUM_V4L2_BUFFERS;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
        perror("VIDIOC_REQBUFS");
        close(fd);
        return -1;
    }

    if (req.count < NUM_V4L2_BUFFERS) {
        fprintf(stderr,
                "camera returned only %u buffers\n",
                req.count);
        close(fd);
        return -1;
    }

    for (unsigned i = 0; i < NUM_V4L2_BUFFERS; ++i) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));

        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
            perror("VIDIOC_QUERYBUF");
            close(fd);
            return -1;
        }

        buffers[i].length = buf.length;

        buffers[i].start = mmap(
            NULL,
            buf.length,
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            fd,
            buf.m.offset);

        if (buffers[i].start == MAP_FAILED) {
            perror("mmap camera");
            close(fd);
            return -1;
        }

        if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
            perror("VIDIOC_QBUF");
            close(fd);
            return -1;
        }
    }

    enum v4l2_buf_type type =
        V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if (ioctl(fd, VIDIOC_STREAMON, &type) < 0) {
        perror("VIDIOC_STREAMON");
        close(fd);
        return -1;
    }

    return fd;
}

static void camera_close(
    int fd,
    struct camera_buffer buffers[NUM_V4L2_BUFFERS])
{
    if (fd < 0)
        return;

    enum v4l2_buf_type type =
        V4L2_BUF_TYPE_VIDEO_CAPTURE;

    ioctl(fd, VIDIOC_STREAMOFF, &type);

    for (unsigned i = 0; i < NUM_V4L2_BUFFERS; ++i) {
        if (buffers[i].start &&
            buffers[i].start != MAP_FAILED) {
            munmap(buffers[i].start,
                   buffers[i].length);
        }
    }

    close(fd);
}


/* Landmark-only model: use a centered square ROI, not a person detector.
 * Pose coordinates are in 256x256 ROI pixels; fd 3 retains 640x480 YUYV. */
#define ROI_SIZE CAM_H
#define ROI_X ((CAM_W-ROI_SIZE)/2)
#define ROI_Y 0
#define OUTPUTS 5
static const size_t output_elements[OUTPUTS] = {195, 64*64*39, 1, 128*128, 117};

static void rgb_at(const uint8_t *src, size_t stride, int x, int y, float rgb[3])
{
    const uint8_t *p=src+(size_t)y*stride+(x/2)*4;
    int c=(int)p[(x&1)?2:0]-16, d=(int)p[1]-128, e=(int)p[3]-128;
    if(c<0)c=0;
    rgb[0]=clip_u8((298*c+409*e+128)>>8);
    rgb[1]=clip_u8((298*c-100*d-208*e+128)>>8);
    rgb[2]=clip_u8((298*c+516*d+128)>>8);
}

typedef struct { int x0, x1; float wx; } ResizeX;
typedef struct { int y0, y1; float wy; } ResizeY;
static ResizeX resize_x[NET_W];
static ResizeY resize_y[NET_H];
static uint8_t rgb_crop[ROI_SIZE * ROI_SIZE * 3];
typedef struct { double yuv, resize_quantization; } PreTimes;

static void init_resize_tables(void)
{
    for (int x=0;x<NET_W;++x) {
        float fx=fmaxf(0,fminf(ROI_SIZE-1,(x+0.5f)*ROI_SIZE/NET_W-0.5f));
        int x0=(int)fx;
        resize_x[x]=(ResizeX){x0,x0+1<ROI_SIZE?x0+1:x0,fx-x0};
    }
    for (int y=0;y<NET_H;++y) {
        float fy=fmaxf(0,fminf(ROI_SIZE-1,(y+0.5f)*ROI_SIZE/NET_H-0.5f));
        int y0=(int)fy;
        resize_y[y]=(ResizeY){y0,y0+1<ROI_SIZE?y0+1:y0,fy-y0};
    }
}

#ifdef __ARM_NEON
/* 32-bit products avoid overflow of the BT.601 coefficients. */
static uint8x8_t rgb_channel(int16x8_t c,int16x8_t d,int16x8_t e,int kd,int ke)
{
    int32x4_t lo=vmull_n_s16(vget_low_s16(c),298);
    int32x4_t hi=vmull_n_s16(vget_high_s16(c),298);
    lo=vmlal_n_s16(lo,vget_low_s16(d),kd);
    hi=vmlal_n_s16(hi,vget_high_s16(d),kd);
    lo=vmlal_n_s16(lo,vget_low_s16(e),ke);
    hi=vmlal_n_s16(hi,vget_high_s16(e),ke);
    lo=vshrq_n_s32(vaddq_s32(lo,vdupq_n_s32(128)),8);
    hi=vshrq_n_s32(vaddq_s32(hi,vdupq_n_s32(128)),8);
    return vqmovun_s16(vcombine_s16(vqmovn_s32(lo),vqmovn_s32(hi)));
}
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
static inline int16x8_t quantize8_neon(float32x4_t a, float32x4_t b)
{
    const float32x4_t scale=vdupq_n_f32(32768.0f/255.0f);
    return vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(vmulq_f32(a,scale))),
                       vqmovn_s32(vcvtnq_s32_f32(vmulq_f32(b,scale))));
}
#endif

static void convert_frame(const uint8_t *src, size_t stride, int16_t *dst, PreTimes *times)
{
    double start=now_ms(),converted=0;
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
    #ifdef _OPENMP
    #pragma omp for schedule(static)
    #endif
    for(int y=0;y<ROI_SIZE;++y) {
        int x=0;
        #ifdef __ARM_NEON
        const uint8_t ui[8]={0,0,2,2,4,4,6,6},vi[8]={1,1,3,3,5,5,7,7};
        for(;x+7<ROI_SIZE;x+=8) {
            uint8x8x2_t p=vld2_u8(src+(y+ROI_Y)*stride+(ROI_X+x)*2);
            int16x8_t c=vmaxq_s16(vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(p.val[0])),vdupq_n_s16(16)),vdupq_n_s16(0));
            int16x8_t d=vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vtbl1_u8(p.val[1],vld1_u8(ui)))),vdupq_n_s16(128));
            int16x8_t e=vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vtbl1_u8(p.val[1],vld1_u8(vi)))),vdupq_n_s16(128));
            uint8x8x3_t rgb={{rgb_channel(c,d,e,0,409),rgb_channel(c,d,e,-100,-208),rgb_channel(c,d,e,516,0)}};
            vst3_u8(rgb_crop+(y*ROI_SIZE+x)*3,rgb);
        }
        #endif
        for(;x<ROI_SIZE;++x) {
            float rgb[3];rgb_at(src,stride,ROI_X+x,ROI_Y+y,rgb);
            for(int ch=0;ch<3;++ch)rgb_crop[(y*ROI_SIZE+x)*3+ch]=(uint8_t)rgb[ch];
        }
    }
    /* The first loop's barrier makes the complete RGB crop available. */
    #ifdef _OPENMP
    #pragma omp single
    #endif
    { converted=now_ms(); }
    #ifdef _OPENMP
    #pragma omp for schedule(static)
    #endif
    for(int y=0;y<NET_H;++y) {
        const ResizeY *ry=&resize_y[y];
        float wy=ry->wy;
        for(int base=0;base<NET_W;base+=8) {
            /* Only eight pixels of scratch, private to this worker. */
            float values[24];
            int count=NET_W-base<8?NET_W-base:8;
            for(int k=0;k<count;++k) {
                const ResizeX *rx=&resize_x[base+k];
                float wx=rx->wx;
                const uint8_t *a=rgb_crop+(ry->y0*ROI_SIZE+rx->x0)*3;
                const uint8_t *b=rgb_crop+(ry->y0*ROI_SIZE+rx->x1)*3;
                const uint8_t *c=rgb_crop+(ry->y1*ROI_SIZE+rx->x0)*3;
                const uint8_t *d=rgb_crop+(ry->y1*ROI_SIZE+rx->x1)*3;
                for(int ch=0;ch<3;++ch)
                    values[k*3+ch]=(a[ch]*(1-wx)+b[ch]*wx)*(1-wy)+(c[ch]*(1-wx)+d[ch]*wx)*wy;
            }
            int i=0;
            int16_t *out=dst+(y*NET_W+base)*3;
            #if defined(__aarch64__) && defined(__ARM_NEON)
            for(;i+7<count*3;i+=8)
                vst1q_s16(out+i,quantize8_neon(vld1q_f32(values+i),vld1q_f32(values+i+4)));
            #endif
            for(;i<count*3;++i) {
                long q=lrintf(values[i]*(32768.0f/255.0f));
                out[i]=(int16_t)(q>32767?32767:q);
            }
        }
    }
    }
    times->yuv=converted-start;
    times->resize_quantization=now_ms()-converted;
}

static void pixel(uint8_t *image,int x,int y,int green)
{
    if(x<0||x>=CAM_W||y<0||y>=CAM_H)
        return;
    uint8_t *p=image+((size_t)y*CAM_W+(x&~1))*2;
    p[(x&1)?2:0]=green?145:210; 
    p[1]=green?54:16; 
    p[3]=green?34:146;
}
static void line(uint8_t *image,int x0,int y0,int x1,int y1,int green)
{
    int dx=abs(x1-x0),sx=x0<x1?1:-1,dy=-abs(y1-y0),sy=y0<y1?1:-1,err=dx+dy;
    for(;;) {
        pixel(image,x0,y0,green);
        pixel(image,x0+1,y0,green);
        if(x0==x1&&y0==y1)
            break;
        int e=2*err;
        if(e>=dy){
            err+=dy;
            x0+=sx;
        }
        if(e<=dx){
            err+=dx;
            y0+=sy;
        }
    }
}
static float sigmoid(float x) { return 1.0f/(1.0f+expf(-x)); }
static unsigned draw_pose(uint8_t *video,const float *landmarks,float threshold,int draw)
{
    static const unsigned edges[][2]={
        {0,1},{1,2},{2,3},{3,7},{0,4},{4,5},{5,6},{6,8},{9,10},
        {11,12},{11,13},{13,15},{15,17},{15,19},{15,21},{17,19},
        {12,14},{14,16},{16,18},{16,20},{16,22},{18,20},
        {11,23},{12,24},{23,24},{23,25},{25,27},{27,29},{29,31},{27,31},
        {24,26},{26,28},{28,30},{30,32},{28,32}};
    int xs[33],ys[33],valid[33];
    unsigned visible=0;
    for(int i=0;i<33;++i) {
        const float *p=landmarks+5*i;
        valid[i]=isfinite(p[0]) && isfinite(p[1]) && p[0]>=0 && p[0]<NET_W && p[1]>=0 && p[1]<NET_H &&
                 sigmoid(p[3])>=threshold&&sigmoid(p[4])>=threshold;
        if(!valid[i])
            continue;
        xs[i]=ROI_X+(int)lroundf(p[0]*ROI_SIZE/NET_W);
        ys[i]=ROI_Y+(int)lroundf(p[1]*ROI_SIZE/NET_H);
        ++visible;
    }
    if(!draw)return visible;
    for(unsigned i=0;i<sizeof(edges)/sizeof(edges[0]);++i) {
        unsigned a=edges[i][0],b=edges[i][1];
        if(valid[a]&&valid[b])
            line(video,xs[a],ys[a],xs[b],ys[b],1);
    }
    for(int i=0;i<33;++i){
        if(valid[i]){
            for(int dy=-2;dy<=2;++dy)
                for(int dx=-2;dx<=2;++dx)
                    pixel(video,xs[i]+dx,ys[i]+dy,0);
        }
    }
    return visible;
}
static float option(const char *name,float fallback)
{
    const char *s=getenv(name); if(!s)return fallback;
    char *end;float value=strtof(s,&end);
    if(end==s||*end||!isfinite(value)||value<0||value>1) {
        fprintf(stderr,"Invalid %s; using %.2f\n",name,fallback);return fallback;
    }
    return value;
}
static int tensor_is_int16_dfp(const vip_buffer_create_params_t *p)
{
    return p->data_format==VIP_BUFFER_FORMAT_INT16 &&
           p->quant_format==VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT;
}
static size_t elements(const vip_buffer_create_params_t *p)
{
    size_t n=1;
    for(unsigned i=0;i<p->num_of_dims;++i)
        n*=p->sizes[i];
    return n;
}

int main(int argc,char **argv)
{
    if(argc>3){
        fprintf(stderr,"Usage: %s [network_binary.nb] [/dev/video0]\n",argv[0]);
        return 2;
    }
    int flags=fcntl(3,F_GETFL),streaming=flags>=0 && (flags&O_ACCMODE)!=O_RDONLY;

    struct sigaction sa={0};
    sa.sa_handler=stop_capture;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT,&sa,NULL);
    sigaction(SIGTERM,&sa,NULL);
    signal(SIGPIPE,SIG_IGN);
    const char *model=argc>1 ? argv[1]:NBG_FILE, *camera=argc>2 ? argv[2] : CAMERA_DEVICE;

    float pose_threshold=option("POSE_THRESHOLD",0.5f), landmark_threshold=option("LANDMARK_THRESHOLD",0.5f);

    int rc=1,initialized=0, prepared=0, cam_fd=-1;
    size_t stride=CAM_W*2;
    vip_network net=NULL;
    vip_buffer input=NULL, output[OUTPUTS]={0};
    vip_buffer_create_params_t ip,op[OUTPUTS];

    struct camera_buffer cb[NUM_V4L2_BUFFERS]={{0}};
    uint8_t *video=NULL;
    unsigned processed=0,poses=0;
    double total_npu=0,total_loop=0;
    double total_wait=0,total_pre=0,total_post=0,total_stream=0,total_work=0;
    if(streaming){
        video=malloc(CAM_W*CAM_H*2);
        if(!video)goto cleanup;
    }
    CHECK(vip_init());
    initialized=1;
    CHECK(vip_create_network(model,0,VIP_CREATE_NETWORK_FROM_FILE,&net));
    vip_uint32_t ni=0,no=0;
    CHECK(vip_query_network(net,VIP_NETWORK_PROP_INPUT_COUNT,&ni));
    CHECK(vip_query_network(net,VIP_NETWORK_PROP_OUTPUT_COUNT,&no));
    if(ni!=1||no!=OUTPUTS){
        fprintf(stderr,"Expected one input and five outputs\n");
        goto cleanup;
    }
    print_tensor(net,0,0);
    if(fill_params(net,0,0,&ip))
        goto cleanup;
    if(!tensor_is_int16_dfp(&ip)||ip.quant_data.dfp.fixed_point_pos!=15||
       ip.num_of_dims!=4||ip.sizes[0]!=3||ip.sizes[1]!=NET_W||ip.sizes[2]!=NET_H||ip.sizes[3]!=1) {
        fprintf(stderr,"Expected NHWC 1x256x256x3 INT16 DFP FL15 input\n");
        goto cleanup;
    }
    for(unsigned i=0;i<OUTPUTS;++i) {
        print_tensor(net,1,i);
        if(fill_params(net,1,i,&op[i]))
            goto cleanup;
        if(!tensor_is_int16_dfp(&op[i])||elements(&op[i])!=output_elements[i]) {
            fprintf(stderr,"Unexpected output %u contract (use INT16 BlazePose lite NB)\n",i);
            goto cleanup;
        }
    }
    CHECK(vip_prepare_network(net));
    prepared=1;
    CHECK(vip_create_buffer(&ip,sizeof(ip),&input));
    CHECK(vip_set_input(net,0,input));

    for(unsigned i=0;i<OUTPUTS;++i) {
        CHECK(vip_create_buffer(&op[i],sizeof(op[i]),&output[i]));
        CHECK(vip_set_output(net,i,output[i]));
    }
    init_resize_tables();
    cam_fd=camera_open(camera,cb,&stride);

    if(cam_fd<0)
        goto cleanup;

    fprintf(stderr,"BlazePose live: center %dx%d ROI, NHWC RGB [0,1], INT16 FL15.\n"
                   "Stand inside the crop; this model has no person detector. Ctrl+C to stop.\n"
                   "Pose threshold %.2f; landmark threshold %.2f; fd 3 stream %s.\n",
            ROI_SIZE,ROI_SIZE,pose_threshold,landmark_threshold,streaming?"enabled":"disabled");
            
    while(!stopping) {
        double start=now_ms();
        struct v4l2_buffer vb={0};
        vb.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
        vb.memory=V4L2_MEMORY_MMAP;
        if(ioctl(cam_fd,VIDIOC_DQBUF,&vb)<0){
            if(errno==EINTR)
                continue;
            perror("VIDIOC_DQBUF");
            goto cleanup;
        }
        double captured=now_ms();
        size_t needed=(CAM_H-1)*stride+CAM_W*2;
        if(vb.index>=NUM_V4L2_BUFFERS||vb.bytesused<needed||cb[vb.index].length<needed) {
            fprintf(stderr,"Invalid camera frame\n");
            goto cleanup;
        }
        int16_t *dst=vip_map_buffer(input);
        if(!dst){
            fprintf(stderr,"Input mapping failed\n");
            goto cleanup;
        }
        PreTimes pt;
        convert_frame(cb[vb.index].start,stride,dst,&pt);
        vip_unmap_buffer(input);
        double copy_start=now_ms();
        if(streaming){
            for(unsigned y=0;y<CAM_H;++y)
                memcpy(video+(size_t)y*CAM_W*2,(uint8_t *)cb[vb.index].start+y*stride,CAM_W*2);
        }
        double copy_ms=now_ms()-copy_start;
        if(ioctl(cam_fd,VIDIOC_QBUF,&vb)<0){
            perror("VIDIOC_QBUF");
            goto cleanup;
        }
        if(stopping)
            break;

        double flush_start=now_ms();
        CHECK(vip_flush_buffer(input,VIP_BUFFER_OPER_TYPE_FLUSH));
        double flush_ms=now_ms()-flush_start;
        double t0=now_ms();
        CHECK(vip_run_network(net));
        double inference_done=now_ms();
        double npu=inference_done-t0;
        CHECK(vip_flush_buffer(output[2],VIP_BUFFER_OPER_TYPE_INVALIDATE));
        int16_t *flag=vip_map_buffer(output[2]);

        if(!flag)
            goto cleanup;

        float confidence=ldexpf((float)flag[0],-(int)(int8_t)op[2].quant_data.dfp.fixed_point_pos);
        vip_unmap_buffer(output[2]);unsigned visible=0;
        if(streaming) {
            line(video,ROI_X,0,ROI_X,CAM_H-1,0);
            line(video,ROI_X+ROI_SIZE-1,0,ROI_X+ROI_SIZE-1,CAM_H-1,0);
        }
        if(confidence>=pose_threshold) {
            CHECK(vip_flush_buffer(output[0],VIP_BUFFER_OPER_TYPE_INVALIDATE));
            int16_t *raw=vip_map_buffer(output[0]);
            if(!raw)
                goto cleanup;
            float points[195],step=ldexpf(1.0f,-(int)(int8_t)op[0].quant_data.dfp.fixed_point_pos);
            for(int i=0;i<195;++i)
                points[i]=raw[i]*step;
            vip_unmap_buffer(output[0]);
            visible=draw_pose(video,points,landmark_threshold,streaming);++poses;
        }
        double post_done=now_ms();
        if(streaming && stream_frame(video,CAM_W*2)<0)
            goto cleanup;
        double finished=now_ms(),elapsed=finished-start;
        double wait=captured-start,pre=t0-captured,post=post_done-inference_done;
        double stream=finished-post_done,work=post_done-captured;
        total_wait+=wait;
        total_pre+=pre;
        total_post+=post;
        total_stream+=stream;
        total_work+=work;
        total_npu+=npu;
        total_loop+=elapsed;
        ++processed;
        if(processed<=5||processed%30==0) {
            fprintf(stderr,"  yuv_conversion %.2f ms | resize_quantization %.2f ms | buffer_copy %.2f ms | npu_flush %.2f ms\n",
                    pt.yuv,pt.resize_quantization,copy_ms,flush_ms);
            char temperature[32];
            npu_temperature(temperature, sizeof(temperature));
            fprintf(stderr,"frame %u | wait %.2f | pre %.2f | NPU %.2f | post %.2f | stream %.2f | work %.2f | loop %.2f ms | pose %.4f | landmarks %u/33 | NPU temp %s\n",
                    processed,wait,pre,npu,post,stream,work,elapsed,confidence,visible,temperature);
        }
    }
    rc=0;
cleanup:
    fprintf(stderr,"\nProcessed %u frames (%u poses) | mean NPU %.2f ms | mean loop %.2f ms\n",
            processed,poses,processed?total_npu/processed:0,processed?total_loop/processed:0);
    if(processed)
        fprintf(stderr,"Mean stages (ms): wait %.2f | pre %.2f | post %.2f | stream %.2f | work %.2f\n",
        total_wait/processed,total_pre/processed,total_post/processed,total_stream/processed,total_work/processed);
    fprintf(stderr,"Shutting down camera and NPU resources...\n");
    if(cam_fd>=0)
        camera_close(cam_fd,cb);
    for(unsigned i=0;i<OUTPUTS;++i)
        if(output[i])
            vip_destroy_buffer(output[i]);
    if(input)
        vip_destroy_buffer(input);
    if(prepared&&net)
        vip_finish_network(net);
    if(net)
        vip_destroy_network(net);
    if(initialized)
        vip_destroy();
    free(video);
    if(streaming)
        close(3);
    fprintf(stderr,rc?"BlazePose stopped after an error (exit status 1).\n":"BlazePose stopped successfully (exit status 0).\n");
    return rc;
}
