__global__ void simpleTaskKernel(unsigned meanExp, float* cls,
                              		float* clx, float* cly)
{
  unsigned i= blockDim.x*blockIdx.x+threadIdx.x;
  if(i<meanExp){
    if (cls[i] != 0){
      clx[i] /= cls[i];
      cly[i] /= cls[i];
    }
    cls[i]= 0;
  }
}
