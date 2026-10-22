/* Test-only VIP runtime: output order deliberately layer3, layer1, layer2. */
#include "vip_lite.h"
#include <stdlib.h>
#include <string.h>
struct MockBuffer {void *data;size_t bytes;vip_buffer_create_params_t p;};
struct MockNetwork {vip_buffer input,outs[3];int fail_run,floating,score;};
static struct MockNetwork *last_feature;
int vip_init(void){return 0;} int vip_destroy(void){return 0;}
int vip_create_network(const char *path,unsigned size,int kind,vip_network *net){
 (void)size;(void)kind;struct MockNetwork *n=calloc(1,sizeof(*n));if(!n)return -1;*net=n;
 n->fail_run=strstr(path,"runfail")!=NULL;n->floating=strstr(path,"floating")!=NULL;n->score=strstr(path,"score")!=NULL;
 if(!n->score){last_feature=n;}return 0;
}
int vip_query_network(vip_network n,int prop,void *out){struct MockNetwork *net=n;*(unsigned *)out=prop==VIP_NETWORK_PROP_INPUT_COUNT||net->score?1:3;return 0;}
static int query(struct MockNetwork *n,int output,unsigned port,int prop,void *out){
 unsigned dims[4]={256,256,3,1};
 if(output){unsigned widths[3]={16,64,32},channels[3]={256,64,128};dims[0]=dims[1]=widths[port];dims[2]=channels[port];}
 if(n->score){dims[0]=dims[1]=64;dims[2]=output?1:100;}
 switch(prop){
 case VIP_BUFFER_PROP_NUM_OF_DIMENSION:*(unsigned *)out=4;break;
 case VIP_BUFFER_PROP_SIZES_OF_DIMENSION:memcpy(out,dims,sizeof(dims));break;
 case VIP_BUFFER_PROP_DATA_FORMAT:*(int *)out=n->score?VIP_BUFFER_FORMAT_FP16:output&&n->floating?VIP_BUFFER_FORMAT_FP32:VIP_BUFFER_FORMAT_INT8;break;
 case VIP_BUFFER_PROP_QUANT_FORMAT:*(int *)out=n->score|| (output&&n->floating)?VIP_BUFFER_QUANTIZE_NONE:VIP_BUFFER_QUANTIZE_TF_ASYMM;break;
 case VIP_BUFFER_PROP_TF_SCALE:*(float *)out=output?1.f/128:.02f;break;
 case VIP_BUFFER_PROP_TF_ZERO_POINT:*(uint32_t *)out=output?(uint32_t)-128:(uint32_t)-14;break;
 default:return -1;
 }return 0;
}
int vip_query_input(vip_network n,unsigned i,int p,void *out){return query(n,0,i,p,out);}
int vip_query_output(vip_network n,unsigned i,int p,void *out){return query(n,1,i,p,out);}
int vip_prepare_network(vip_network n){(void)n;return 0;}int vip_finish_network(vip_network n){(void)n;return 0;}int vip_destroy_network(vip_network n){if(n==last_feature)last_feature=NULL;free(n);return 0;}
int vip_create_buffer(vip_buffer_create_params_t *p,size_t size,vip_buffer *out){
 (void)size;*out=calloc(1,sizeof(**out));if(!*out)return -1;(*out)->p=*p;
 size_t count=1;for(unsigned i=0;i<p->num_of_dims;i++)count*=p->sizes[i];
 (*out)->bytes=count*(p->data_format==VIP_BUFFER_FORMAT_FP32?4:p->data_format==VIP_BUFFER_FORMAT_FP16?2:1);(*out)->data=calloc(1,(*out)->bytes);return (*out)->data?0:-1;
}
size_t vip_get_buffer_size(vip_buffer b){return b->bytes;}int vip_destroy_buffer(vip_buffer b){free(b->data);free(b);return 0;}
int vip_set_input(vip_network n,unsigned i,vip_buffer b){(void)i;((struct MockNetwork *)n)->input=b;return 0;}
int vip_set_output(vip_network n,unsigned i,vip_buffer b){((struct MockNetwork *)n)->outs[i]=b;return 0;}
void *vip_map_buffer(vip_buffer b){return b->data;}int vip_unmap_buffer(vip_buffer b){(void)b;return 0;}int vip_flush_buffer(vip_buffer b,int op){(void)b;(void)op;return 0;}
int vip_run_network(vip_network n){
 struct MockNetwork *net=n;if(net->fail_run)return -99;
 if(net->score){
  /* Validate half feature channel order; return d2=4, or infinity fixture. */
  const uint16_t *x=net->input->data;
  if(x[0]!=0x3800u || x[4096]!=0x3c00u || x[8192]!=0x3e00u)return -88;
  uint16_t *y=net->outs[0]->data;
  for(int i=0;i<4096;i++)y[i]=net->floating?0x7c00u:0x4400u;
  return 0;
 }
 for(unsigned port=0;port<3;port++){
  int layer=port==0?2:port==1?0:1;vip_buffer b=net->outs[port];size_t count=b->bytes/(net->floating?4:1);
  for(size_t i=0;i<count;i++){
   int q=-64+layer*64+(int)(i%13);
   if(net->floating)((float *)b->data)[i]=(q+128)/128.f;
   else ((int8_t *)b->data)[i]=(int8_t)q;
  }
 }return 0;
}
float mock_input(unsigned i){return ((int8_t *)last_feature->input->data)[i];}
