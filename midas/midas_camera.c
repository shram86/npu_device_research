/* MiDaS live depth fusion; V4L2/VIPLite/fd3 pipeline adapted from yolo_camera.c. */
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

#include <vip_lite.h>
#include <signal.h>
#include <poll.h>

#define NBG_FILE "./network_binary_int16.nb"
#define CAMERA_DEVICE "/dev/video0"
#define CAM_W 640
#define CAM_H 480
#define NUM_V4L2_BUFFERS 4
#define MAX_NET 384

struct camera_buffer {
    void *start;
    size_t length;
};

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

static int fill_params(vip_network net, int output, uint32_t index,
                       vip_buffer_create_params_t *p);

static int print_tensor(vip_network net, int output, uint32_t index)
{
    vip_buffer_create_params_t p;
    if (fill_params(net, output, index, &p)) {
        fprintf(stderr, "Failed to query %s %u parameters\n", output ? "output" : "input", index);
        return -1;
    }
    /* Query directly into the SDK structure: no guessed name/shape array sizes. */
    fprintf(stderr, "%s %u: format=%d quant=%d dims=",
            output ? "output" : "input", index, (int)p.data_format, (int)p.quant_format);
    for (uint32_t i=0; i<p.num_of_dims; ++i)
        fprintf(stderr, "%u%s", (unsigned)p.sizes[i], i+1==p.num_of_dims ? "" : "x");
    if (p.quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT)
        fprintf(stderr, " DFP=%d", (int)(int8_t)p.quant_data.dfp.fixed_point_pos);
    else if (p.quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM)
        fprintf(stderr, " scale=%.9g zp=%d", p.quant_data.affine.scale,
                (int)p.quant_data.affine.zeroPoint);
    fprintf(stderr, "\n");
    return 0;
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

    if (dims == 0 || dims > sizeof(p->sizes) / sizeof(p->sizes[0])) {
        fprintf(stderr, "Invalid SDK dimension count: %u\n", (unsigned)dims);
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
        goto fail;
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
        fprintf(stderr, "Camera requires 640x480 YUYV with valid stride\n");
        goto fail;
    }
    *stride = fmt.fmt.pix.bytesperline;

    struct v4l2_streamparm parm;
    memset(&parm, 0, sizeof(parm));

    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = 30;

    if (ioctl(fd, VIDIOC_S_PARM, &parm) < 0) {
        perror("VIDIOC_S_PARM");
        goto fail;
    }

    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));

    req.count = NUM_V4L2_BUFFERS;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
        perror("VIDIOC_REQBUFS");
        goto fail;
    }

    if (req.count < NUM_V4L2_BUFFERS) {
        fprintf(stderr,
                "camera returned only %u buffers\n",
                req.count);
        goto fail;
    }

    for (unsigned i = 0; i < NUM_V4L2_BUFFERS; ++i) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));

        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
            perror("VIDIOC_QUERYBUF");
            goto fail;
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
            goto fail;
        }

        if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
            perror("VIDIOC_QBUF");
            goto fail;
        }
    }

    enum v4l2_buf_type type =
        V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if (ioctl(fd, VIDIOC_STREAMON, &type) < 0) {
        perror("VIDIOC_STREAMON");
        goto fail;
    }

    return fd;
fail:
    for (unsigned i=0; i<NUM_V4L2_BUFFERS; ++i) {
        if (buffers[i].start && buffers[i].start != MAP_FAILED)
            munmap(buffers[i].start, buffers[i].length);
        buffers[i].start = NULL;
    }
    close(fd);
    return -1;
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

static void fill_rect_yuyv(
    uint8_t *frame,
    int width,
    int height,
    int x1,
    int y1,
    int x2,
    int y2,
    uint8_t y_value)
{
    if (x1 < 0) x1 = 0;
    if (y1 < 0) y1 = 0;
    if (x2 >= width)  x2 = width - 1;
    if (y2 >= height) y2 = height - 1;

    for (int y = y1; y <= y2; ++y) {
        for (int x = x1; x <= x2; ++x) {
            frame[y * width * 2 + x * 2] = y_value;
        }
    }
}

/*
 * Minimal 5x7 font.
 * Each byte is one row; low 5 bits are pixels.
 * Letters, digits, spaces and decimal points for the live status labels.
 */

static const uint8_t FONT_SPACE[7] = {
    0,0,0,0,0,0,0
};

static const uint8_t FONT_DOT[7] = {
    0,0,0,0,0,0x0C,0x0C
};

static const uint8_t FONT_DIGITS[10][7] = {
    {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}, /* 0 */
    {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}, /* 1 */
    {0x0E,0x11,0x01,0x02,0x04,0x08,0x1F}, /* 2 */
    {0x1E,0x01,0x01,0x0E,0x01,0x01,0x1E}, /* 3 */
    {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}, /* 4 */
    {0x1F,0x10,0x10,0x1E,0x01,0x01,0x1E}, /* 5 */
    {0x0E,0x10,0x10,0x1E,0x11,0x11,0x0E}, /* 6 */
    {0x1F,0x01,0x02,0x04,0x08,0x08,0x08}, /* 7 */
    {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}, /* 8 */
    {0x0E,0x11,0x11,0x0F,0x01,0x01,0x0E}  /* 9 */
};

static const uint8_t FONT_ALPHA[26][7] = {
    {0x0E,0x11,0x11,0x1F,0x11,0x11,0x11}, /* A */
    {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E}, /* B */
    {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E}, /* C */
    {0x1E,0x11,0x11,0x11,0x11,0x11,0x1E}, /* D */
    {0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F}, /* E */
    {0x1F,0x10,0x10,0x1E,0x10,0x10,0x10}, /* F */
    {0x0E,0x11,0x10,0x17,0x11,0x11,0x0F}, /* G */
    {0x11,0x11,0x11,0x1F,0x11,0x11,0x11}, /* H */
    {0x0E,0x04,0x04,0x04,0x04,0x04,0x0E}, /* I */
    {0x07,0x02,0x02,0x02,0x02,0x12,0x0C}, /* J */
    {0x11,0x12,0x14,0x18,0x14,0x12,0x11}, /* K */
    {0x10,0x10,0x10,0x10,0x10,0x10,0x1F}, /* L */
    {0x11,0x1B,0x15,0x15,0x11,0x11,0x11}, /* M */
    {0x11,0x19,0x15,0x13,0x11,0x11,0x11}, /* N */
    {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E}, /* O */
    {0x1E,0x11,0x11,0x1E,0x10,0x10,0x10}, /* P */
    {0x0E,0x11,0x11,0x11,0x15,0x12,0x0D}, /* Q */
    {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11}, /* R */
    {0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E}, /* S */
    {0x1F,0x04,0x04,0x04,0x04,0x04,0x04}, /* T */
    {0x11,0x11,0x11,0x11,0x11,0x11,0x0E}, /* U */
    {0x11,0x11,0x11,0x11,0x11,0x0A,0x04}, /* V */
    {0x11,0x11,0x11,0x15,0x15,0x15,0x0A}, /* W */
    {0x11,0x11,0x0A,0x04,0x0A,0x11,0x11}, /* X */
    {0x11,0x11,0x0A,0x04,0x04,0x04,0x04}, /* Y */
    {0x1F,0x01,0x02,0x04,0x08,0x10,0x1F}  /* Z */
};

static const uint8_t *font_char(char ch)
{
    if (ch >= 'a' && ch <= 'z')
        ch = (char)(ch - 'a' + 'A');

    if (ch >= 'A' && ch <= 'Z')
        return FONT_ALPHA[ch - 'A'];

    if (ch >= '0' && ch <= '9')
        return FONT_DIGITS[ch - '0'];

    if (ch == '.')
        return FONT_DOT;

    return FONT_SPACE;
}

static void draw_char_yuyv(
    uint8_t *frame,
    int width,
    int height,
    int x,
    int y,
    char ch,
    int scale)
{
    const uint8_t *glyph = font_char(ch);

    for (int gy = 0; gy < 7; ++gy) {
        for (int gx = 0; gx < 5; ++gx) {

            if (!(glyph[gy] & (1 << (4 - gx))))
                continue;

            for (int sy = 0; sy < scale; ++sy) {
                for (int sx = 0; sx < scale; ++sx) {

                    int px = x + gx * scale + sx;
                    int py = y + gy * scale + sy;

                    if (px >= 0 && px < width &&
                        py >= 0 && py < height) {

                        /* Bright text: modify Y only. */
                        frame[py * width * 2 + px * 2] = 235;
                    }
                }
            }
        }
    }
}

static void draw_text_yuyv(
    uint8_t *frame,
    int width,
    int height,
    int x,
    int y,
    const char *text,
    int scale)
{
    while (*text) {
        draw_char_yuyv(
            frame,
            width,
            height,
            x,
            y,
            *text,
            scale
        );

        x += 6 * scale;
        ++text;
    }
}

static volatile sig_atomic_t stopping;
static void stop_capture(int signal_number)
{
    (void)signal_number;
    stopping = 1;
}

/* Pillow-style antialiased separable bicubic, including byte clipping between passes. */
typedef struct { int start, count; float weight[32]; } Filter;
static Filter resize_x[MAX_NET], resize_y[MAX_NET];
static uint8_t rgb[CAM_W * CAM_H * 3], horizontal[MAX_NET * CAM_H * 3];
static uint8_t video[CAM_W * CAM_H * 2];
static float depth[MAX_NET * MAX_NET];

static float cubic(float x)
{
    x=fabsf(x);
    if (x<1) return ((1.5f*x-2.5f)*x)*x+1;
    if (x<2) return ((-0.5f*x+2.5f)*x-4)*x+2;
    return 0;
}

static int filters(Filter *table, int source, int destination)
{
    float scale=(float)source/destination, support=2*fmaxf(1,scale);
    for(int i=0;i<destination;++i) {
        float center=(i+0.5f)*scale, total=0;
        int begin=(int)(center-support+0.5f), end=(int)(center+support+0.5f);
        if(begin<0)begin=0;
        if(end>source)end=source;
        Filter *f=&table[i]; f->start=begin;f->count=end-begin;
        if(f->count<1 || f->count>32)return -1;
        for(int j=0;j<f->count;++j) {
            f->weight[j]=cubic((begin+j-center+0.5f)/fmaxf(1,scale));
            total+=f->weight[j];
        }
        for(int j=0;j<f->count;++j)f->weight[j]/=total;
    }
    return 0;
}

static int tensor_bytes(const vip_buffer_create_params_t *p)
{
    if(p->data_format==VIP_BUFFER_FORMAT_INT8 || p->data_format==VIP_BUFFER_FORMAT_UINT8)return 1;
    if(p->data_format==VIP_BUFFER_FORMAT_INT16)return 2;
    if(p->data_format==VIP_BUFFER_FORMAT_FP32)return 4;
    return 0;
}

static size_t elements(const vip_buffer_create_params_t *p)
{
    size_t n=1;
    for(unsigned i=0;i<p->num_of_dims;++i)n*=p->sizes[i];
    return n;
}

static int encoding(const vip_buffer_create_params_t *p, float *scale, int *zp)
{
    *zp=0;
    if(!tensor_bytes(p))return -1;
    if(p->data_format==VIP_BUFFER_FORMAT_FP32) {
        *scale=1;return p->quant_format==VIP_BUFFER_QUANTIZE_NONE?0:-1;
    }
    if(p->quant_format==VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT)
        *scale=ldexpf(1,-(int)(int8_t)p->quant_data.dfp.fixed_point_pos);
    else if(p->quant_format==VIP_BUFFER_QUANTIZE_TF_ASYMM) {
        *scale=p->quant_data.affine.scale;*zp=p->quant_data.affine.zeroPoint;
    } else return -1;
    return isfinite(*scale)&&*scale>0?0:-1;
}

static void store(void *dst, size_t i, float real,
                  const vip_buffer_create_params_t *p, float scale, int zp)
{
    if(p->data_format==VIP_BUFFER_FORMAT_FP32){((float *)dst)[i]=real;return;}
    float q=nearbyintf(real/scale)+zp;
    if(p->data_format==VIP_BUFFER_FORMAT_INT8)
        ((int8_t *)dst)[i]=(int8_t)fmaxf(-128,fminf(127,q));
    else if(p->data_format==VIP_BUFFER_FORMAT_UINT8)
        ((uint8_t *)dst)[i]=(uint8_t)fmaxf(0,fminf(255,q));
    else ((int16_t *)dst)[i]=(int16_t)fmaxf(-32768,fminf(32767,q));
}

static void prepare_frame(const uint8_t *source, size_t stride, void *dst,
                          int nw,int nh,const vip_buffer_create_params_t *p,float scale,int zp)
{
    const float mean[3]={0.485f,0.456f,0.406f},std[3]={0.229f,0.224f,0.225f};
    for(int y=0;y<CAM_H;++y) {
        memcpy(video+(size_t)y*CAM_W*2,source+(size_t)y*stride,CAM_W*2);
        for(int x=0;x<CAM_W;++x) {
            const uint8_t *pair=source+(size_t)y*stride+(x/2)*4;
            int c=pair[(x&1)?2:0]-16,d=pair[1]-128,e=pair[3]-128;
            if(c<0)c=0;
            uint8_t *pixel=rgb+((size_t)y*CAM_W+x)*3;
            pixel[0]=clip_u8((298*c+409*e+128)>>8);
            pixel[1]=clip_u8((298*c-100*d-208*e+128)>>8);
            pixel[2]=clip_u8((298*c+516*d+128)>>8);
        }
    }
    #pragma omp parallel for schedule(static)
    for(int y=0;y<CAM_H;++y)for(int x=0;x<nw;++x)for(int c=0;c<3;++c) {
        Filter *f=&resize_x[x];float sum=0;
        for(int j=0;j<f->count;++j)sum+=f->weight[j]*rgb[((size_t)y*CAM_W+f->start+j)*3+c];
        horizontal[((size_t)y*nw+x)*3+c]=clip_u8((int)floorf(sum+0.5f));
    }
    #pragma omp parallel for schedule(static)
    for(int y=0;y<nh;++y)for(int x=0;x<nw;++x)for(int c=0;c<3;++c) {
        Filter *f=&resize_y[y];float sum=0;
        for(int j=0;j<f->count;++j)sum+=f->weight[j]*horizontal[((size_t)(f->start+j)*nw+x)*3+c];
        float pixel=clip_u8((int)floorf(sum+0.5f));
        store(dst,(size_t)c*nw*nh+(size_t)y*nw+x,(pixel/255-mean[c])/std[c],p,scale,zp);
    }
}

static float sample_depth(int x,int y,int width,int height)
{
    float fx=fmaxf(0,(x+0.5f)*width/CAM_W-0.5f),fy=fmaxf(0,(y+0.5f)*height/CAM_H-0.5f);
    int x0=(int)fx,y0=(int)fy,x1=x0+1<width?x0+1:x0,y1=y0+1<height?y0+1:y0;
    float wx=fx-x0,wy=fy-y0;
    return (1-wy)*((1-wx)*depth[y0*width+x0]+wx*depth[y0*width+x1])+
           wy*((1-wx)*depth[y1*width+x0]+wx*depth[y1*width+x1]);
}

static void color(float t,int *y,int *u,int *v)
{
    /* Blue -> cyan -> yellow -> red: increasing inverse depth means nearer. */
    float position=fmaxf(0,fminf(1,t))*3;
    int segment=(int)position;float f=position-segment;
    int r,g,b;
    if(segment==0){r=0;g=(int)(255*f);b=255;}
    else if(segment==1){r=(int)(255*f);g=255;b=(int)(255*(1-f));}
    else {r=255;g=(int)(255*(1-(position-2)));b=0;}
    *y=clip_u8(((66*r+129*g+25*b+128)>>8)+16);
    *u=clip_u8(((-38*r-74*g+112*b+128)>>8)+128);
    *v=clip_u8(((112*r-94*g-18*b+128)>>8)+128);
}

static int fuse(const void *output,const vip_buffer_create_params_t *p,
                int width,int height,float scale,int zp,float alpha,float *low,float *high)
{
    float minimum=INFINITY,maximum=-INFINITY;
    for(int i=0;i<width*height;++i) {
        float q;
        if(p->data_format==VIP_BUFFER_FORMAT_INT8)q=((const int8_t *)output)[i];
        else if(p->data_format==VIP_BUFFER_FORMAT_UINT8)q=((const uint8_t *)output)[i];
        else if(p->data_format==VIP_BUFFER_FORMAT_INT16)q=((const int16_t *)output)[i];
        else q=((const float *)output)[i];
        depth[i]=(q-zp)*scale;
        if(!isfinite(depth[i]))return -1;
        if(depth[i]<minimum)minimum=depth[i];
        if(depth[i]>maximum)maximum=depth[i];
    }
    if(!isfinite(*low)){*low=minimum;*high=maximum;}
    else {*low=0.9f * (*low)+0.1f*minimum;*high=0.9f * (*high)+0.1f*maximum;}
    float span=fmaxf(*high-*low,1e-6f);
    #pragma omp parallel for schedule(static)
    for(int y=0;y<CAM_H;++y)for(int x=0;x<CAM_W;x+=2) {
        int y0,u0,v0,y1,u1,v1;
        color((sample_depth(x,y,width,height)-*low)/span,&y0,&u0,&v0);
        color((sample_depth(x+1,y,width,height)-*low)/span,&y1,&u1,&v1);
        uint8_t *pair=video+((size_t)y*CAM_W+x)*2;
        pair[0]=clip_u8((int)lroundf((1-alpha)*pair[0]+alpha*y0));
        pair[2]=clip_u8((int)lroundf((1-alpha)*pair[2]+alpha*y1));
        pair[1]=clip_u8((int)lroundf((1-alpha)*pair[1]+alpha*(u0+u1)*0.5f));
        pair[3]=clip_u8((int)lroundf((1-alpha)*pair[3]+alpha*(v0+v1)*0.5f));
    }
    return 0;
}

static int stream_frame(void)
{
    size_t sent=0,total=sizeof(video);
    while(sent<total && !stopping) {
        ssize_t n=write(3,video+sent,total-sent);
        if(n<0 && errno==EINTR)continue;
        if(n<0 && errno==EPIPE){stopping=1;return 0;}
        if(n<=0){perror("video fd3 write");return -1;}
        sent+=(size_t)n;
    }
    return 0;
}

int main(int argc,char **argv)
{
    vip_network net=NULL;vip_buffer input=NULL,output=NULL;
    int initialized=0,prepared=0,cam_fd=-1,rc=1;
    void *input_mapping=NULL,*output_mapping=NULL;
    struct camera_buffer buffers[NUM_V4L2_BUFFERS]={{0}};
    size_t stride=0;
    const char *model=NBG_FILE,*camera=CAMERA_DEVICE;
    const char *input_type="auto";
    float alpha=0.5f;
    int positional=0;
    for(int i=1;i<argc;++i) {
        if(strcmp(argv[i],"--input-type")==0) {
            if(++i>=argc){fprintf(stderr,"--input-type requires auto, int8 or int16\n");return 1;}
            input_type=argv[i];
            if(strcmp(input_type,"auto") && strcmp(input_type,"int8") && strcmp(input_type,"int16")) {
                fprintf(stderr,"--input-type requires auto, int8 or int16\n");return 1;
            }
        } else if(strcmp(argv[i],"--help")==0) {
            fprintf(stderr,"Usage: %s [model.nb] [/dev/video0] [alpha 0..1] [--input-type auto|int8|int16]\n",argv[0]);
            return 0;
        } else if(argv[i][0]=='-') {
            fprintf(stderr,"Unknown option: %s\n",argv[i]);return 1;
        } else if(positional==0){model=argv[i];++positional;}
        else if(positional==1){camera=argv[i];++positional;}
        else if(positional==2) {
            char *end;alpha=strtof(argv[i],&end);
            if(end==argv[i] || *end || !isfinite(alpha) || alpha<0 || alpha>1) {
                fprintf(stderr,"Alpha must be 0..1\n");return 1;
            }
            ++positional;
        } else {
            fprintf(stderr,"Too many positional arguments; use --help\n");return 1;
        }
    }
    if(fcntl(3,F_GETFD)<0){fprintf(stderr,"Open fd3 for YUYV streaming (see midas_camera.md)\n");return 1;}
    struct sigaction action={0};action.sa_handler=stop_capture;sigemptyset(&action.sa_mask);
    sigaction(SIGINT,&action,NULL);sigaction(SIGTERM,&action,NULL);
    signal(SIGPIPE,SIG_IGN);
    CHECK(vip_init());initialized=1;
    CHECK(vip_create_network(model,0,VIP_CREATE_NETWORK_FROM_FILE,&net));
    vip_uint32_t ni=0,no=0;
    CHECK(vip_query_network(net,VIP_NETWORK_PROP_INPUT_COUNT,&ni));
    CHECK(vip_query_network(net,VIP_NETWORK_PROP_OUTPUT_COUNT,&no));
    if(ni!=1 || no!=1){fprintf(stderr,"Expected one RGB input and one depth output\n");goto cleanup;}
    if(print_tensor(net,0,0) || print_tensor(net,1,0))goto cleanup;
    vip_buffer_create_params_t ip,op;
    if(fill_params(net,0,0,&ip)||fill_params(net,1,0,&op))goto cleanup;
    if((strcmp(input_type,"int8")==0 && ip.data_format!=VIP_BUFFER_FORMAT_INT8) ||
       (strcmp(input_type,"int16")==0 && ip.data_format!=VIP_BUFFER_FORMAT_INT16)) {
        fprintf(stderr,"Requested %s input does not match loaded NB format %d. Use the matching NB or --input-type auto.\n",
                input_type,(int)ip.data_format);
        goto cleanup;
    }
    float input_scale,output_scale;int input_zp,output_zp;
    if(encoding(&ip,&input_scale,&input_zp)||encoding(&op,&output_scale,&output_zp)) {
        fprintf(stderr,"Unsupported IO encoding; use INT8/UINT8/INT16 DFP or affine, or FP32\n");goto cleanup;
    }
    int nw=(int)ip.sizes[0],nh=(int)ip.sizes[1],dw=(int)op.sizes[0],dh=(int)op.sizes[1];
    if(ip.num_of_dims!=4 || ip.sizes[2]!=3 || ip.sizes[3]!=1 ||
       (nw!=256 && nw!=384) || nh!=nw ||
       (op.num_of_dims!=3 && op.num_of_dims!=4) || dw!=nw || dh!=nh ||
       elements(&op)!=(size_t)dw*dh) {
        fprintf(stderr,"Expected WHCN RGB 256/384 square and a matching single depth plane\n");goto cleanup;
    }
    if(filters(resize_x,CAM_W,nw)||filters(resize_y,CAM_H,nh))goto cleanup;
    CHECK(vip_prepare_network(net));prepared=1;
    CHECK(vip_create_buffer(&ip,sizeof(ip),&input));
    CHECK(vip_create_buffer(&op,sizeof(op),&output));
    if(vip_get_buffer_size(input)!=elements(&ip)*(size_t)tensor_bytes(&ip) ||
       vip_get_buffer_size(output)!=elements(&op)*(size_t)tensor_bytes(&op)) {
        fprintf(stderr,"Unexpected padded tensor storage\n");goto cleanup;
    }
    CHECK(vip_set_input(net,0,input));CHECK(vip_set_output(net,0,output));
    cam_fd=camera_open(camera,buffers,&stride);if(cam_fd<0)goto cleanup;
    fprintf(stderr,"MiDaS fusion alpha=%.2f; blue=far red=near (relative inverse depth). Ctrl-C stops.\n",alpha);
    float low=NAN,high=NAN;unsigned frame=0;
    while(!stopping) {
        double t0=now_ms();
        struct pollfd waitfd={cam_fd,POLLIN,0};
        int ready=poll(&waitfd,1,1000);
        if(ready<0 && errno==EINTR)continue;
        if(ready<0){perror("camera poll");goto cleanup;}
        if(!ready)continue;
        if(!(waitfd.revents&POLLIN)){fprintf(stderr,"Camera poll failed\n");goto cleanup;}
        struct v4l2_buffer vbuf={0};vbuf.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;vbuf.memory=V4L2_MEMORY_MMAP;
        if(ioctl(cam_fd,VIDIOC_DQBUF,&vbuf)<0) {
            if(errno==EINTR)continue;
            perror("VIDIOC_DQBUF");goto cleanup;
        }
        size_t required=(CAM_H-1)*stride+CAM_W*2;
        if(vbuf.index>=NUM_V4L2_BUFFERS || vbuf.bytesused<required ||
           buffers[vbuf.index].length<required){fprintf(stderr,"Invalid camera frame\n");goto cleanup;}
        if(vbuf.flags&V4L2_BUF_FLAG_ERROR) {
            if(ioctl(cam_fd,VIDIOC_QBUF,&vbuf)<0)goto cleanup;
            continue;
        }
        double t1=now_ms();
        input_mapping=vip_map_buffer(input);if(!input_mapping)goto cleanup;
        prepare_frame(buffers[vbuf.index].start,stride,input_mapping,nw,nh,&ip,input_scale,input_zp);
        vip_unmap_buffer(input);input_mapping=NULL;
        if(ioctl(cam_fd,VIDIOC_QBUF,&vbuf)<0){perror("VIDIOC_QBUF");goto cleanup;}
        CHECK(vip_flush_buffer(input,VIP_BUFFER_OPER_TYPE_FLUSH));
        double t2=now_ms();
        CHECK(vip_run_network(net));double t3=now_ms();
        CHECK(vip_flush_buffer(output,VIP_BUFFER_OPER_TYPE_INVALIDATE));
        output_mapping=vip_map_buffer(output);if(!output_mapping)goto cleanup;
        if(fuse(output_mapping,&op,dw,dh,output_scale,output_zp,alpha,&low,&high)) {
            fprintf(stderr,"Non-finite depth output\n");goto cleanup;
        }
        vip_unmap_buffer(output);output_mapping=NULL;
        char label[96];snprintf(label,sizeof(label),"MIDAS RELATIVE DEPTH  NPU %.1f MS",t3-t2);
        fill_rect_yuyv(video,CAM_W,CAM_H,0,0,strlen(label)*6+8,14,25);
        draw_text_yuyv(video,CAM_W,CAM_H,4,4,label,1);
        draw_text_yuyv(video,CAM_W,CAM_H,8,CAM_H-18,"BLUE FAR     RED NEAR",1);
        double t4=now_ms();
        if(stream_frame())goto cleanup;
        double t5=now_ms();
        fprintf(stderr,"frame %u | wait %.2f pre %.2f NPU %.2f fusion %.2f stream %.2f ms | range %.1f..%.1f\n",
                ++frame,t1-t0,t2-t1,t3-t2,t4-t3,t5-t4,low,high);
    }
    rc=0;
cleanup:
    if(input_mapping)vip_unmap_buffer(input);
    if(output_mapping)vip_unmap_buffer(output);
    camera_close(cam_fd,buffers);
    if(output)vip_destroy_buffer(output);
    if(input)vip_destroy_buffer(input);
    if(prepared)vip_finish_network(net);
    if(net)vip_destroy_network(net);
    if(initialized)vip_destroy();
    return rc;
}
