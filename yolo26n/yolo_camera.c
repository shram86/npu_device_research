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
#include <arm_neon.h>

#define NBG_FILE   "./yolo_direct_calibrate.nb"
#define INPUT_FILE "./input_0.dat"

#define CAMERA_DEVICE "/dev/video0"
#define CAM_W 640
#define CAM_H 480
#define NET_W 640
#define NET_H 640
#define PAD_Y 80
#define NUM_V4L2_BUFFERS 4

#define YOLO_POSITIONS 8400
#define YOLO_CLASSES   80

#define MAX_CANDIDATES 8400
#define NMS_IOU_THRESHOLD 0.45f

static const char *COCO_NAMES[80] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

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
 *   RGB CHW 640x640
 *   signed INT8 affine:
 *
 *       real = (q - (-128)) * (1/255)
 *
 * Therefore for an RGB byte p:
 *
 *       q = p - 128
 */
static void yuyv_to_npu_input(
    const uint8_t *src,
    int8_t *dst)
{
    const size_t plane = NET_W * NET_H;

    int8_t *r_plane = dst;
    int8_t *g_plane = dst + plane;
    int8_t *b_plane = dst + 2 * plane;

    /*
     * YOLO letterbox padding value = 114.
     * Quantized INT8 representation = 114 - 128 = -14.
     */
    memset(dst, (uint8_t)(int8_t)-14, 3 * plane);

    for (int y = 0; y < CAM_H; ++y) {
        const uint8_t *row = src + y * CAM_W * 2;
        const int out_y = y + PAD_Y;

        for (int x = 0; x < CAM_W; x += 2) {
            int Y0 = row[0];
            int U  = row[1];
            int Y1 = row[2];
            int V  = row[3];
            row += 4;

            /*
             * Integer BT.601 limited-range YUV -> RGB.
             *
             * R = 1.164*(Y-16) + 1.596*(V-128)
             * G = 1.164*(Y-16) - 0.392*(U-128)
             *                    - 0.813*(V-128)
             * B = 1.164*(Y-16) + 2.017*(U-128)
             */

            int d = U - 128;
            int e = V - 128;

            int c0 = Y0 - 16;
            int c1 = Y1 - 16;

            if (c0 < 0) c0 = 0;
            if (c1 < 0) c1 = 0;

            int r0 = (298 * c0 + 409 * e + 128) >> 8;
            int g0 = (298 * c0 - 100 * d - 208 * e + 128) >> 8;
            int b0 = (298 * c0 + 516 * d + 128) >> 8;

            int r1 = (298 * c1 + 409 * e + 128) >> 8;
            int g1 = (298 * c1 - 100 * d - 208 * e + 128) >> 8;
            int b1 = (298 * c1 + 516 * d + 128) >> 8;

            size_t i0 = (size_t)out_y * NET_W + x;
            size_t i1 = i0 + 1;

            r_plane[i0] = (int8_t)((int)clip_u8(r0) - 128);
            g_plane[i0] = (int8_t)((int)clip_u8(g0) - 128);
            b_plane[i0] = (int8_t)((int)clip_u8(b0) - 128);

            r_plane[i1] = (int8_t)((int)clip_u8(r1) - 128);
            g_plane[i1] = (int8_t)((int)clip_u8(g1) - 128);
            b_plane[i1] = (int8_t)((int)clip_u8(b1) - 128);
        }
    }
}

static int camera_open(
    const char *device,
    struct camera_buffer buffers[NUM_V4L2_BUFFERS])
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

typedef struct {
    float x1, y1, x2, y2;
    int16_t score_raw;
    int class_id;
    int position;
} DetectionCandidate;

static float box_iou(const DetectionCandidate *a,
                     const DetectionCandidate *b)
{
    float ix1 = a->x1 > b->x1 ? a->x1 : b->x1;
    float iy1 = a->y1 > b->y1 ? a->y1 : b->y1;
    float ix2 = a->x2 < b->x2 ? a->x2 : b->x2;
    float iy2 = a->y2 < b->y2 ? a->y2 : b->y2;

    float iw = ix2 - ix1;
    float ih = iy2 - iy1;

    if (iw <= 0.0f || ih <= 0.0f)
        return 0.0f;

    float intersection = iw * ih;

    float area_a =
        (a->x2 - a->x1) * (a->y2 - a->y1);

    float area_b =
        (b->x2 - b->x1) * (b->y2 - b->y1);

    float union_area = area_a + area_b - intersection;

    if (union_area <= 0.0f)
        return 0.0f;

    return intersection / union_area;
}

static int compare_candidate_score(const void *pa, const void *pb)
{
    const DetectionCandidate *a =
        (const DetectionCandidate *)pa;
    const DetectionCandidate *b =
        (const DetectionCandidate *)pb;

    if (a->score_raw < b->score_raw)
        return 1;

    if (a->score_raw > b->score_raw)
        return -1;

    return 0;
}

static void draw_rect_yuyv(
    uint8_t *frame,
    int width,
    int height,
    int x1,
    int y1,
    int x2,
    int y2)
{
    /* Clip to camera frame. */
    if (x1 < 0) x1 = 0;
    if (y1 < 0) y1 = 0;
    if (x2 >= width)  x2 = width - 1;
    if (y2 >= height) y2 = height - 1;

    if (x2 <= x1 || y2 <= y1)
        return;

    const int thickness = 3;

    /*
     * YUYV:
     *
     * pixel 0: Y at byte 0
     * pixel 1: Y at byte 2
     *
     * Therefore Y for pixel (x,y):
     * frame[y * width * 2 + x * 2]
     *
     * We modify only Y, leaving U/V untouched.
     */
    for (int t = 0; t < thickness; ++t) {

        int top = y1 + t;
        int bottom = y2 - t;

        if (top >= 0 && top < height) {
            for (int x = x1; x <= x2; ++x)
                frame[top * width * 2 + x * 2] = 235;
        }

        if (bottom >= 0 && bottom < height) {
            for (int x = x1; x <= x2; ++x)
                frame[bottom * width * 2 + x * 2] = 235;
        }

        int left = x1 + t;
        int right = x2 - t;

        if (left >= 0 && left < width) {
            for (int y = y1; y <= y2; ++y)
                frame[y * width * 2 + left * 2] = 235;
        }

        if (right >= 0 && right < width) {
            for (int y = y1; y <= y2; ++y)
                frame[y * width * 2 + right * 2] = 235;
        }
    }
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
 * Only characters needed for "bottle 0.00" initially.
 */

static const uint8_t FONT_SPACE[7] = {
    0,0,0,0,0,0,0
};

static const uint8_t FONT_B[7] = {
    0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E
};

static const uint8_t FONT_O[7] = {
    0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E
};

static const uint8_t FONT_T[7] = {
    0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04
};

static const uint8_t FONT_L[7] = {
    0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F
};

static const uint8_t FONT_E[7] = {
    0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F
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

int main(void)
{
    int cam_fd = -1;
    double avg_scan = 0.0;
    int avg_count = 0;

    struct camera_buffer cam_buffers[NUM_V4L2_BUFFERS];
    memset(cam_buffers, 0, sizeof(cam_buffers));

    vip_network net = NULL;
    vip_buffer input = NULL;
    vip_buffer out0 = NULL;
    vip_buffer out1 = NULL;

    FILE *f = NULL;
    int initialized = 0;
    int prepared = 0;
    int rc = 1;

    vip_uint32_t input_count = 0;
    vip_uint32_t output_count = 0;

    fprintf(stderr, "VIPLite version: 0x%08x\n", vip_get_version());

    CHECK(vip_init());
    initialized = 1;

    CHECK(vip_create_network(
        NBG_FILE,
        0,
        VIP_CREATE_NETWORK_FROM_FILE,
        &net));

    CHECK(vip_query_network(
        net, VIP_NETWORK_PROP_INPUT_COUNT, &input_count));
    CHECK(vip_query_network(
        net, VIP_NETWORK_PROP_OUTPUT_COUNT, &output_count));

    fprintf(stderr, "network inputs=%u outputs=%u\n",
           input_count, output_count);

    for (uint32_t i = 0; i < input_count; ++i)
        print_tensor(net, 0, i);

    for (uint32_t i = 0; i < output_count; ++i)
        print_tensor(net, 1, i);

    if (input_count != 1 || output_count != 2) {
        fprintf(stderr, "unexpected network I/O count\n");
        goto cleanup;
    }

    CHECK(vip_prepare_network(net));
    prepared = 1;

    vip_buffer_create_params_t ip, op0, op1;

    if (fill_params(net, 0, 0, &ip) ||
        fill_params(net, 1, 0, &op0) ||
        fill_params(net, 1, 1, &op1)) {
        fprintf(stderr, "failed to build buffer parameters\n");
        goto cleanup;
    }

    CHECK(vip_create_buffer(&ip, sizeof(ip), &input));
    CHECK(vip_create_buffer(&op0, sizeof(op0), &out0));
    CHECK(vip_create_buffer(&op1, sizeof(op1), &out1));

    cam_fd = camera_open(CAMERA_DEVICE, cam_buffers);
    if (cam_fd < 0)
        goto cleanup;

    fprintf(stderr, "buffer sizes: input=%u out0=%u out1=%u bytes\n",
           vip_get_buffer_size(input),
           vip_get_buffer_size(out0),
           vip_get_buffer_size(out1));

    CHECK(vip_set_input(net, 0, input));
    CHECK(vip_set_output(net, 0, out0));
    CHECK(vip_set_output(net, 1, out1));

    fprintf(stderr, "\nStarting live inference...\n");
    fprintf(stderr, "Press Ctrl-C to stop.\n\n");

    for (unsigned frame = 0; frame < 600; ++frame) {

        struct v4l2_buffer vbuf;
        memset(&vbuf, 0, sizeof(vbuf));

        vbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        vbuf.memory = V4L2_MEMORY_MMAP;

        /* Wait for next camera frame. */
        if (ioctl(cam_fd, VIDIOC_DQBUF, &vbuf) < 0) {
            perror("VIDIOC_DQBUF");
            goto cleanup;
        }

        if (frame == 0) {
            fprintf(stderr,
                    "camera frame: bytesused=%u expected=%u\n",
                    vbuf.bytesused,
                    CAM_W * CAM_H * 2);
        }

        double t0 = now_ms();

        /*
        * VIP input is INT8:
        *
        * RGB uint8 0..255
        *       ↓
        * q = RGB - 128
        *       ↓
        * INT8 -128..127
        */
        int8_t *in_ptr = (int8_t *)vip_map_buffer(input);

        if (!in_ptr) {
            fprintf(stderr, "vip_map_buffer(input) failed\n");
            goto cleanup;
        }

        yuyv_to_npu_input(
            (const uint8_t *)cam_buffers[vbuf.index].start,
            in_ptr);

        /*
        * First-frame sanity check.
        */
        if (frame == 0) {
            size_t n = vip_get_buffer_size(input);

            int min_q = 127;
            int max_q = -128;

            for (size_t i = 0; i < n; ++i) {
                int q = in_ptr[i];

                if (q < min_q) min_q = q;
                if (q > max_q) max_q = q;
            }

            fprintf(stderr, "INT8 input sanity: min=%d max=%d\n",
                min_q, max_q);

            fprintf(stderr, "Expected range: within [-128,127]\n");
            fprintf(stderr, "Letterbox INT8 value: -14\n\n");
        }

        vip_unmap_buffer(input);

        double t1 = now_ms();

        CHECK(vip_flush_buffer(
            input,
            VIP_BUFFER_OPER_TYPE_FLUSH));

        double t2 = now_ms();

        CHECK(vip_run_network(net));

        double t3 = now_ms();

        double tp0 = now_ms();

        CHECK(vip_flush_buffer(
            out1,
            VIP_BUFFER_OPER_TYPE_INVALIDATE));

        CHECK(vip_flush_buffer(
            out0,
            VIP_BUFFER_OPER_TYPE_INVALIDATE));

        double tp1 = now_ms();

        int16_t *boxes = (int16_t *)vip_map_buffer(out0);

        int16_t *scores = (int16_t *)vip_map_buffer(out1);

        double tp2 = now_ms();

        if (!boxes || !scores) {
            fprintf(stderr, "map outputs failed\n");
            goto cleanup;
        }

        int16_t global_max = 0;

        size_t candidates = 0;

        const int16_t threshold_raw = 8192;  /* 0.25 * 32768 */


        DetectionCandidate dets[MAX_CANDIDATES];
        /* Include setup in scan+decode so POST accounts for all scan time. */
        double ts0 = tp2;
        /*
        * Step 1: find the best class for each of the 8400 positions.
        * Keep positions whose best class score >= 0.25.
        */
        #define BLOCK_P 256

        int16_t best_scores[YOLO_POSITIONS];
        int16_t best_classes[YOLO_POSITIONS];

        #pragma omp parallel for schedule(static)
        for (int base = 0; base < YOLO_POSITIONS; base += BLOCK_P) {

            int end = base + BLOCK_P;
            if (end > YOLO_POSITIONS)
                end = YOLO_POSITIONS;

            /* Initialize this position block. */
            for (int p = base; p < end; ++p) {
                best_scores[p] = 0;
                best_classes[p] = -1;
            }

            /*
            * Class-major traversal:
            * scores for p=base..end are contiguous.
            */
            for (int c = 0; c < YOLO_CLASSES; ++c) {

                const int16_t *row =
                    scores + c * YOLO_POSITIONS;

                int p = base;

                /* Process 8 positions per iteration. */
                for (; p + 7 < end; p += 8) {

                    int16x8_t old_score =
                        vld1q_s16(&best_scores[p]);

                    int16x8_t new_score =
                        vld1q_s16(&row[p]);

                    /*
                    * mask lane = 0xffff when new_score > old_score,
                    * otherwise 0x0000.
                    *
                    * Strict > preserves our original tie behaviour.
                    */
                    uint16x8_t mask =
                        vcgtq_s16(new_score, old_score);

                    int16x8_t selected_score =
                        vbslq_s16(mask, new_score, old_score);

                    vst1q_s16(&best_scores[p], selected_score);

                    int16x8_t old_class =
                        vld1q_s16(&best_classes[p]);

                    int16x8_t new_class =
                        vdupq_n_s16((int16_t)c);

                    int16x8_t selected_class =
                        vbslq_s16(mask, new_class, old_class);

                    vst1q_s16(&best_classes[p], selected_class);
                }

                /* Tail, because 8400/block boundaries need not be multiples of 8. */
                for (; p < end; ++p) {

                    int16_t q = row[p];

                    if (q > best_scores[p]) {
                        best_scores[p] = q;
                        best_classes[p] = (int16_t)c;
                    }
                }
            }
        }

        for (int p = 0; p < YOLO_POSITIONS; ++p) {

            int16_t best_score = best_scores[p];
            int best_class = best_classes[p];

            if (best_score > global_max)
                global_max = best_score;

            if (best_score < threshold_raw)
                continue;

            if (candidates >= MAX_CANDIDATES)
                break;

            /*
            * boxes output is DFP16 FL=5:
            * real_value = raw / 32.
            *
            * Layout:
            * [cx(8400), cy(8400), w(8400), h(8400)]
            */
            float cx =
                boxes[0 * YOLO_POSITIONS + p] / 32.0f;

            float cy =
                boxes[1 * YOLO_POSITIONS + p] / 32.0f;

            float w =
                boxes[2 * YOLO_POSITIONS + p] / 32.0f;

            float h =
                boxes[3 * YOLO_POSITIONS + p] / 32.0f;

            DetectionCandidate *d = &dets[candidates++];

            d->x1 = cx - 0.5f * w;
            d->y1 = cy - 0.5f * h;
            d->x2 = cx + 0.5f * w;
            d->y2 = cy + 0.5f * h;

            d->score_raw = best_score;
            d->class_id = best_class;
            d->position = p;
        }

        double ts1 = now_ms();

        /*
        * Step 2: highest confidence first.
        */
        qsort(
            dets,
            candidates,
            sizeof(DetectionCandidate),
            compare_candidate_score
        );

        double ts2 = now_ms();

        /*
        * Step 3: class-aware NMS.
        *
        * suppressed[i] means a higher-confidence box of the
        * SAME class overlapped this one by >= 0.45 IoU.
        */
        unsigned char suppressed[MAX_CANDIDATES] = {0};
        size_t final_detections = 0;

        for (size_t i = 0; i < candidates; ++i) {

            if (suppressed[i])
                continue;

            DetectionCandidate *keep = &dets[i];

            ++final_detections;

            /*
            * Convert from 640x640 network coordinates back to
            * our 640x480 camera coordinates.
            *
            * We padded 80 pixels at the top and bottom.
            */
            float x1 = keep->x1;
            float y1 = keep->y1 - PAD_Y;
            float x2 = keep->x2;
            float y2 = keep->y2 - PAD_Y;

            /*
            * Clip only for display/camera coordinates.
            */
            if (x1 < 0.0f) x1 = 0.0f;
            if (y1 < 0.0f) y1 = 0.0f;
            if (x2 > CAM_W - 1) x2 = CAM_W - 1;
            if (y2 > CAM_H - 1) y2 = CAM_H - 1;

            fprintf(
                stderr,
                "  DET class=%d conf=%.3f "
                "box=[%.1f %.1f %.1f %.1f]\n",
                keep->class_id,
                keep->score_raw / 32768.0f,
                x1, y1, x2, y2
            );
            draw_rect_yuyv(
                (uint8_t *)cam_buffers[vbuf.index].start,
                CAM_W,
                CAM_H,
                (int)x1,
                (int)y1,
                (int)x2,
                (int)y2
                );

           const char *class_name = "unknown";

            if (keep->class_id >= 0 && keep->class_id < 80)
            {
                class_name = COCO_NAMES[keep->class_id];
            }

            char label[64];

            snprintf(
                label,
                sizeof(label),
                "%s %.2f",
                class_name,
                keep->score_raw / 32768.0f
            );

            uint8_t *video_frame =
                (uint8_t *)cam_buffers[vbuf.index].start;

            int text_scale = 1;

            int box_x1 = (int)x1;
            int box_x2 = (int)x2;
            int box_y1 = (int)y1;

            /*
            * 5 pixels per glyph + 1 pixel spacing.
            * draw_text advances 6 pixels per character.
            */
            int text_w =
                (int)strlen(label) * 6 * text_scale;

            /* 7-pixel glyph + 2 pixels padding each side. */
            int label_h =
                7 * text_scale + 4;

            /*
            * Header should never be narrower than the detection box,
            * but grows when the text requires more space.
            */
            int box_w = box_x2 - box_x1 + 1;
            int label_w = text_w + 4;

            if (label_w < box_w)
                label_w = box_w;

            int label_x1 = box_x1;
            int label_x2 = label_x1 + label_w - 1;

            /*
            * Don't let a long label run beyond the right edge.
            * Shift the whole label left while keeping its width.
            */
            if (label_x2 >= CAM_W) {
                int shift = label_x2 - (CAM_W - 1);

                label_x1 -= shift;
                label_x2 -= shift;

                if (label_x1 < 0)
                    label_x1 = 0;
            }

            int label_y1 = box_y1 - label_h;
            int label_y2;

            if (label_y1 >= 0) {
                /* Normal case: header directly above box. */
                label_y2 = box_y1 - 1;
            }
            else {
                /* Box touches top: put header inside box. */
                label_y1 = box_y1;
                label_y2 = label_y1 + label_h - 1;
            }

            fill_rect_yuyv(
                video_frame,
                CAM_W,
                CAM_H,
                label_x1,
                label_y1,
                label_x2,
                label_y2,
                25
            );

            draw_text_yuyv(
                video_frame,
                CAM_W,
                CAM_H,
                label_x1 + 2,
                label_y1 + 2,
                label,
                text_scale
            );

            for (size_t j = i + 1; j < candidates; ++j) {

                if (suppressed[j])
                    continue;

                /*
                * Class-aware: boxes belonging to different
                * classes do not suppress one another.
                */
                if (dets[j].class_id != keep->class_id)
                    continue;

                float iou = box_iou(keep, &dets[j]);

                if (iou >= NMS_IOU_THRESHOLD)
                    suppressed[j] = 1;
            }
        }

        float max_score = global_max / 32768.0f;

        double ts3 = now_ms();
        double tp3 = ts3;

        vip_unmap_buffer(out1);
        vip_unmap_buffer(out0);

        double tp4 = now_ms();

        /*
        * Send annotated YUYV frame to FFmpeg.
        */
        size_t video_bytes = vbuf.bytesused;

        double tv0 = now_ms();

        uint8_t *video_frame =
            (uint8_t *)cam_buffers[vbuf.index].start;

        size_t sent = 0;

        while (sent < video_bytes) {

            ssize_t n = write(
                3,
                video_frame + sent,
                video_bytes - sent
            );

            if (n < 0) {
                if (errno == EINTR)
                    continue;

                perror("video FD3 write");
                goto cleanup;
            }

            sent += (size_t)n;
        }

        double tv1 = now_ms();

        fprintf(stderr,
        "VIDEO WRITE | %6.2f ms | %zu bytes\n",
        tv1 - tv0,
        video_bytes);

        /*
        * Return camera buffer immediately.
        */
        if (ioctl(cam_fd, VIDIOC_QBUF, &vbuf) < 0) {
            perror("VIDIOC_QBUF");
            goto cleanup;
        }

        fprintf(stderr,
                "POST | scan+decode %6.2f | sort %6.3f | nms+draw %6.2f ms\n",
                ts1 - ts0,
                ts2 - ts1,
                ts3 - ts2);

       fprintf(stderr,
            "frame %3u | prep %5.2f | NPU %5.2f | "
            "inv %5.2f | map %5.2f | scan %5.2f | unmap %5.2f | "
            "cand %3zu | det %2zu | max %.3f\n",
            frame + 1,
            t1 - t0,
            t3 - t2,
            tp1 - tp0,
            tp2 - tp1,
            tp3 - tp2,
            tp4 - tp3,
            candidates,
            final_detections,
    max_score);
        avg_scan += tp3 - tp2;
        ++avg_count;
    }

    fprintf(stderr, "\nAverage scan+decode time: %.2f ms\n",
            avg_scan / avg_count);




cleanup:

    if (cam_fd >= 0)
        camera_close(cam_fd, cam_buffers);
        
    if (f)
        fclose(f);

    if (out1)
        vip_destroy_buffer(out1);
    if (out0)
        vip_destroy_buffer(out0);
    if (input)
        vip_destroy_buffer(input);

    if (prepared && net)
        vip_finish_network(net);

    if (net)
        vip_destroy_network(net);

    if (initialized)
        vip_destroy();

    return rc;
}
