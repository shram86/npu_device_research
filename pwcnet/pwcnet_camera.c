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

#include <vip_lite.h>
#include <arm_neon.h>

#define NBG_FILE "./network_binary_mixed.nb"
#define CAMERA_DEVICE "/dev/video0"
#define CAM_W 640
#define CAM_H 480
#define NET_W 128
#define NET_H 128
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
        static const char interrupt[] = "\nCtrl+C received: stopping PWC-Net...\n";
        static const char terminate[] = "\nSIGTERM received: stopping PWC-Net...\n";
        if (signo == SIGINT)
            (void)write(STDERR_FILENO, interrupt, sizeof(interrupt) - 1);
        else
            (void)write(STDERR_FILENO, terminate, sizeof(terminate) - 1);
    }
    stopping = 1;
    errno = saved_errno;
}

static int stream_frame(const uint8_t *src, size_t stride)
{
    for (unsigned y = 0; y < CAM_H; ++y) {
        const uint8_t *p = src + y * stride;
        size_t remaining = CAM_W * 2;
        while (remaining && !stopping) {
            ssize_t n = write(3, p, remaining);
            if (n < 0 && errno == EINTR) continue;
            if (n < 0 && errno == EPIPE) { stopping = 1; return 0; }
            if (n <= 0) { perror("write video fd 3"); return -1; }
            p += n; remaining -= (size_t)n;
        }
        if (stopping) break;
    }
    return 0;
}

/* Heatmap range is camera pixels per processed frame pair, not pixels/sec. */
static float heatmap_max = 8.0f;
static float heatmap_alpha = 0.60f;

static float heatmap_option(const char *name, float fallback, float low, float high)
{
    const char *value = getenv(name);
    if (!value) return fallback;
    char *end;
    float parsed = strtof(value, &end);
    if (end == value || *end || !isfinite(parsed) || parsed < low || parsed > high) {
        fprintf(stderr, "Invalid %s=%s; using %g\n", name, value, fallback);
        return fallback;
    }
    return parsed;
}

static void draw_flow(uint8_t *image, const int8_t *flow, float scale, int32_t zp)
{
    float magnitude[NET_W * NET_H];
    const int plane = NET_W * NET_H;
    /* Flow is located on the previous frame. Use actual camera-pixel units,
     * without the old arrow-length amplification. */
    for (int k = 0; k < plane; ++k) {
        float dx = ((int)flow[k] - zp) * scale * ((float)CAM_W / NET_W);
        float dy = ((int)flow[plane+k] - zp) * scale * ((float)CAM_H / NET_H);
        float m = hypotf(dx, dy);
        magnitude[k] = isfinite(m) ? m : 0;
    }
    /* Blue -> cyan -> yellow -> red, BT.601 limited-range YUV.
     * Opacity increases with motion so stationary regions stay untouched. */
    uint8_t colors[256][4];
    for (int i = 0; i < 256; ++i) {
        float t = i / 255.0f;
        float r, g, b;
        if (t < 1.0f/3) { r=0; g=3*t; b=1; }
        else if (t < 2.0f/3) { r=3*t-1; g=1; b=2-3*t; }
        else { r=1; g=3-3*t; b=0; }
        colors[i][0] = (uint8_t)lroundf(16 + 65.481f*r + 128.553f*g + 24.966f*b);
        colors[i][1] = (uint8_t)lroundf(128 - 37.797f*r - 74.203f*g + 112*b);
        colors[i][2] = (uint8_t)lroundf(128 + 112*r - 93.786f*g - 18.214f*b);
        colors[i][3] = (uint8_t)lroundf(255 * heatmap_alpha * t);
    }
    for (int y = 0; y < CAM_H; ++y) {
        float fy = fmaxf(0, fminf(NET_H-1, (y+0.5f)*NET_H/CAM_H-0.5f));
        int y0 = (int)fy, y1 = y0+1 < NET_H ? y0+1 : y0;
        float wy = fy-y0;
        for (int x = 0; x < CAM_W; x += 2) {
            /* Sample at the YUYV pair center; shared U/V are blended once. */
            float fx = fmaxf(0, fminf(NET_W-1, (x+1.0f)*NET_W/CAM_W-0.5f));
            int x0 = (int)fx, x1 = x0+1 < NET_W ? x0+1 : x0;
            float wx = fx-x0;
            float top = magnitude[y0*NET_W+x0]*(1-wx) + magnitude[y0*NET_W+x1]*wx;
            float bot = magnitude[y1*NET_W+x0]*(1-wx) + magnitude[y1*NET_W+x1]*wx;
            float m = top*(1-wy) + bot*wy;
            int level = (int)lroundf(255 * fminf(1, m/heatmap_max));
            const uint8_t *c = colors[level];
            int alpha = c[3], inverse = 255-alpha;
            uint8_t *p = image + ((size_t)y*CAM_W+x)*2;
            p[0] = (uint8_t)((p[0]*inverse+c[0]*alpha+127)/255);
            p[2] = (uint8_t)((p[2]*inverse+c[0]*alpha+127)/255);
            p[1] = (uint8_t)((p[1]*inverse+c[1]*alpha+127)/255);
            p[3] = (uint8_t)((p[3]*inverse+c[2]*alpha+127)/255);
        }
    }
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
        fprintf(stderr, " DFP=%u", fp);
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

/*
 * Camera:
 *   YUYV 640x480
 *   BT.601 limited range
 *
 * Network:
 *   RGB CHW 128x128
 *   signed INT8 affine:
 *
 *       real = (q - (-128)) * (1/255)
 *
 * Therefore for an RGB byte p:
 *
 *       q = p - 128
 */
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


/* Nearest-neighbor full-frame resize, BT.601 limited-range YUYV to RGB. */
static void convert_frame(const uint8_t *src, size_t stride, int8_t *dst)
{
    const int plane = NET_W * NET_H;
    for (int y = 0; y < NET_H; ++y) {
        int sy = y * CAM_H / NET_H;
        const uint8_t *row = src + (size_t)sy * stride;
        for (int x = 0; x < NET_W; ++x) {
            int sx = x * CAM_W / NET_W;
            const uint8_t *p = row + (sx / 2) * 4;
            int Y = p[(sx & 1) ? 2 : 0];
            int d = (int)p[1] - 128, e = (int)p[3] - 128;
            int c = Y - 16;
            if (c < 0) c = 0;
            int r = (298*c + 409*e + 128) >> 8;
            int g = (298*c - 100*d - 208*e + 128) >> 8;
            int b = (298*c + 516*d + 128) >> 8;
            int pos = y * NET_W + x;
            dst[pos] = (int8_t)((int)clip_u8(r) - 128);
            dst[plane + pos] = (int8_t)((int)clip_u8(g) - 128);
            dst[2*plane + pos] = (int8_t)((int)clip_u8(b) - 128);
        }
    }
}

int main(void)
{
    /* Check before SDK/camera opens can claim an otherwise unused fd 3. */
    int flags = fcntl(3, F_GETFL);
    int streaming = flags >= 0 && (flags & O_ACCMODE) != O_RDONLY;
    size_t stride = CAM_W * 2;
    struct sigaction sa = {0};
    sa.sa_handler = stop_capture;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);
    int rc = 1, initialized = 0, prepared = 0, cam_fd = -1;
    vip_network net = NULL;
    vip_buffer in[2] = {NULL, NULL}, out = NULL;
    struct camera_buffer cb[NUM_V4L2_BUFFERS] = {{0}};
    vip_uint32_t ni = 0, no = 0;
    vip_float_t out_scale = 0;
    vip_uint32_t out_zp_u = 0;
    vip_enum out_fmt = 0, out_quant = 0;
    int8_t previous[3 * NET_W * NET_H];
    int have_previous = 0;
    double total_npu = 0, total_loop = 0;
    unsigned processed = 0;
    uint8_t *video_previous = NULL, *video_current = NULL;
    const char *overlay_env = getenv("PWC_FLOW_OVERLAY");
    int overlay = !overlay_env || strcmp(overlay_env, "0") != 0;
    heatmap_max = heatmap_option("PWC_HEATMAP_MAX", 8.0f, 0.01f, 10000.0f);
    heatmap_alpha = heatmap_option("PWC_HEATMAP_ALPHA", 0.60f, 0.0f, 1.0f);
    if (streaming) {
        video_previous = malloc(CAM_W * CAM_H * 2);
        video_current = malloc(CAM_W * CAM_H * 2);
        if (!video_previous || !video_current) {
            fprintf(stderr, "Video buffer allocation failed\n"); goto cleanup;
        }
        fprintf(stderr, "Motion heatmap %s: red >= %g camera px/pair, max opacity %.0f%%.\n",
                overlay ? "enabled" : "disabled", heatmap_max, heatmap_alpha*100);
    }

    CHECK(vip_init()); initialized = 1;
    CHECK(vip_create_network(NBG_FILE, 0, VIP_CREATE_NETWORK_FROM_FILE, &net));
    CHECK(vip_query_network(net, VIP_NETWORK_PROP_INPUT_COUNT, &ni));
    CHECK(vip_query_network(net, VIP_NETWORK_PROP_OUTPUT_COUNT, &no));
    fprintf(stderr, "network inputs=%u outputs=%u\n", ni, no);
    if (ni != 2 || no != 1) goto cleanup;
    for (unsigned i = 0; i < 2; ++i) print_tensor(net, 0, i);
    print_tensor(net, 1, 0);
    CHECK(vip_query_output(net, 0, VIP_BUFFER_PROP_DATA_FORMAT, &out_fmt));
    CHECK(vip_query_output(net, 0, VIP_BUFFER_PROP_QUANT_FORMAT, &out_quant));
    if (out_quant != VIP_BUFFER_QUANTIZE_TF_ASYMM) {
        fprintf(stderr, "Unsupported output quantization\n"); goto cleanup;
    }
    CHECK(vip_query_output(net, 0, VIP_BUFFER_PROP_TF_SCALE, &out_scale));
    CHECK(vip_query_output(net, 0, VIP_BUFFER_PROP_TF_ZERO_POINT, &out_zp_u));
    /* The validated NBG uses signed INT8. Check byte size as a safeguard. */
    CHECK(vip_prepare_network(net)); prepared = 1;
    for (unsigned i = 0; i < 2; ++i) {
        vip_buffer_create_params_t p;
        if (fill_params(net, 0, i, &p)) goto cleanup;
        CHECK(vip_create_buffer(&p, sizeof(p), &in[i]));
        if (vip_get_buffer_size(in[i]) != sizeof(previous)) {
            fprintf(stderr, "Unexpected input buffer size %u\n", vip_get_buffer_size(in[i]));
            goto cleanup;
        }
        CHECK(vip_set_input(net, i, in[i]));
    }
    vip_buffer_create_params_t op;
    if (fill_params(net, 1, 0, &op)) goto cleanup;
    CHECK(vip_create_buffer(&op, sizeof(op), &out));
    if (vip_get_buffer_size(out) != 2 * NET_W * NET_H) {
        fprintf(stderr, "Expected signed INT8 output, got %u bytes\n", vip_get_buffer_size(out));
        goto cleanup;
    }
    CHECK(vip_set_output(net, 0, out));
    cam_fd = camera_open(CAMERA_DEVICE, cb, &stride);
    if (cam_fd < 0) goto cleanup;
    fprintf(stderr, streaming ? "Streaming 640x480 YUYV on fd 3 until interrupted.\n"
                              : "Starting PWC-Net live test (600 captured frames).\n");

    for (unsigned frame = 0; !stopping && (streaming || frame < 600); ++frame) {
        double start = now_ms();
        struct v4l2_buffer vb = {0};
        vb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        vb.memory = V4L2_MEMORY_MMAP;
        if (ioctl(cam_fd, VIDIOC_DQBUF, &vb) < 0) {
            if (errno == EINTR) { if (stopping) break; else continue; }
            perror("VIDIOC_DQBUF"); goto cleanup;
        }
        size_t required = (CAM_H - 1) * stride + CAM_W * 2;
        if (vb.index >= NUM_V4L2_BUFFERS || vb.bytesused < required ||
            cb[vb.index].length < required) {
            fprintf(stderr, "Camera returned an invalid or short frame\n"); goto cleanup;
        }
        int8_t current[sizeof(previous)];
        convert_frame((const uint8_t *)cb[vb.index].start, stride, current);
        if (streaming) {
            const uint8_t *src = cb[vb.index].start;
            for (unsigned y = 0; y < CAM_H; ++y)
                memcpy(video_current + (size_t)y * CAM_W * 2,
                       src + y * stride, CAM_W * 2);
        }
        if (ioctl(cam_fd, VIDIOC_QBUF, &vb) < 0) { perror("VIDIOC_QBUF"); goto cleanup; }
        if (stopping) break;
        if (!have_previous) {
            memcpy(previous, current, sizeof(previous));
            if (streaming) {
                uint8_t *tmp = video_previous;
                video_previous = video_current; video_current = tmp;
            }
            have_previous = 1;
            continue;
        }
        for (unsigned i = 0; i < 2; ++i) {
            int8_t *ptr = (int8_t *)vip_map_buffer(in[i]);
            if (!ptr) { fprintf(stderr, "input map failed\n"); goto cleanup; }
            memcpy(ptr, i ? current : previous, sizeof(previous));
            vip_unmap_buffer(in[i]);
            CHECK(vip_flush_buffer(in[i], VIP_BUFFER_OPER_TYPE_FLUSH));
        }
        double t0 = now_ms();
        CHECK(vip_run_network(net));
        double t1 = now_ms();
        CHECK(vip_flush_buffer(out, VIP_BUFFER_OPER_TYPE_INVALIDATE));
        int8_t *flow = (int8_t *)vip_map_buffer(out);
        if (!flow) { fprintf(stderr, "output map failed\n"); goto cleanup; }
        double sumx = 0, sumy = 0, summag = 0;
        const int plane = NET_W * NET_H;
        /* Central 96x96 crop reduces boundary effects. */
        for (int y = 16; y < 112; ++y) for (int x = 16; x < 112; ++x) {
            int k = y * NET_W + x;
            float dx = ((int)flow[k] - (int32_t)out_zp_u) * out_scale;
            float dy = ((int)flow[plane + k] - (int32_t)out_zp_u) * out_scale;
            sumx += dx; sumy += dy;
            summag += sqrtf(dx*dx + dy*dy);
        }
        if (streaming && overlay)
            draw_flow(video_previous, flow, out_scale, (int32_t)out_zp_u);
        vip_unmap_buffer(out);
        if (streaming) {
            if (stream_frame(overlay ? video_previous : video_current, CAM_W * 2) < 0)
                goto cleanup;
            uint8_t *tmp = video_previous;
            video_previous = video_current; video_current = tmp;
        }
        memcpy(previous, current, sizeof(previous));
        double elapsed = now_ms() - start;
        total_npu += t1-t0; total_loop += elapsed;
        ++processed;
        if (processed <= 5 || processed % 30 == 0) {
            fprintf(stderr, "pair %3u | NPU %.2f ms | loop %.2f ms | "
                    "flow dx=%+.3f dy=%+.3f mag=%.3f px\n",
                    processed, t1-t0, elapsed,
                    sumx/9216.0, sumy/9216.0, summag/9216.0);
        }
    }
    fprintf(stderr, "Processed %u pairs | mean NPU %.2f ms | mean loop %.2f ms\n",
            processed, processed ? total_npu/processed : 0, processed ? total_loop/processed : 0);
    rc = 0;
cleanup:
    fprintf(stderr, "Shutting down camera and NPU resources...\n");
    free(video_previous);
    free(video_current);
    if (cam_fd >= 0) camera_close(cam_fd, cb);
    if (out) vip_destroy_buffer(out);
    for (unsigned i = 0; i < 2; ++i) if (in[i]) vip_destroy_buffer(in[i]);
    if (prepared && net) vip_finish_network(net);
    if (net) vip_destroy_network(net);
    if (initialized) vip_destroy();
    if (streaming) close(3);
    fprintf(stderr, rc == 0 ? "PWC-Net stopped successfully (exit status 0).\n"
                            : "PWC-Net stopped after an error (exit status 1).\n");
    fflush(stderr);
    return rc;
}
