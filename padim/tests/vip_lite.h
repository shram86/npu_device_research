/* Minimal test double, NOT a vendor header and never used in the device build. */
#ifndef TEST_VIP_LITE_H
#define TEST_VIP_LITE_H
#include <stdint.h>
#include <stddef.h>
typedef uint32_t vip_uint32_t;
typedef int32_t vip_int32_t;
typedef uint8_t vip_uint8_t;
typedef float vip_float_t;
typedef int vip_enum;
typedef int vip_status_e;
typedef void *vip_network;
typedef struct MockBuffer *vip_buffer;
typedef struct {
    unsigned num_of_dims, sizes[6];
    int data_format, quant_format, memory_type;
    union { struct { uint8_t fixed_point_pos; } dfp;
            struct { float scale; int32_t zeroPoint; } affine; } quant_data;
} vip_buffer_create_params_t;
enum { VIP_SUCCESS=0, VIP_BUFFER_FORMAT_FP32=1, VIP_BUFFER_FORMAT_INT16,
       VIP_BUFFER_FORMAT_INT8, VIP_BUFFER_FORMAT_UINT8, VIP_BUFFER_FORMAT_FP16 };
enum { VIP_BUFFER_QUANTIZE_NONE=0, VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT,
       VIP_BUFFER_QUANTIZE_TF_ASYMM };
enum { VIP_BUFFER_PROP_NUM_OF_DIMENSION, VIP_BUFFER_PROP_SIZES_OF_DIMENSION,
       VIP_BUFFER_PROP_DATA_FORMAT, VIP_BUFFER_PROP_QUANT_FORMAT,
       VIP_BUFFER_PROP_FIXED_POINT_POS, VIP_BUFFER_PROP_TF_SCALE, VIP_BUFFER_PROP_TF_ZERO_POINT };
enum { VIP_NETWORK_PROP_INPUT_COUNT, VIP_NETWORK_PROP_OUTPUT_COUNT };
enum { VIP_BUFFER_MEMORY_TYPE_DEFAULT, VIP_CREATE_NETWORK_FROM_FILE,
       VIP_BUFFER_OPER_TYPE_FLUSH, VIP_BUFFER_OPER_TYPE_INVALIDATE };
int vip_init(void); int vip_destroy(void);
int vip_create_network(const char *,unsigned,int,vip_network *);
int vip_query_network(vip_network,int,void *);
int vip_query_input(vip_network,unsigned,int,void *);
int vip_query_output(vip_network,unsigned,int,void *);
int vip_prepare_network(vip_network); int vip_finish_network(vip_network); int vip_destroy_network(vip_network);
int vip_create_buffer(vip_buffer_create_params_t *,size_t,vip_buffer *);
size_t vip_get_buffer_size(vip_buffer); int vip_destroy_buffer(vip_buffer);
int vip_set_input(vip_network,unsigned,vip_buffer); int vip_set_output(vip_network,unsigned,vip_buffer);
void *vip_map_buffer(vip_buffer); int vip_unmap_buffer(vip_buffer); int vip_flush_buffer(vip_buffer,int);
int vip_run_network(vip_network);
float mock_input(unsigned);
#endif
