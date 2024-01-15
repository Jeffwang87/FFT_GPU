#include<stdio.h>
#include<math.h>
#include "timerc.h"
#define PI 3.142857
const double pi = 22.0 / 7.0;


#define gerror(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void warm_up_gpu(){
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float a, b;
  a = b = 0.0f;
  b += a + tid;
}


__host__ __device__ int bitrev(int input, int numbits)
{
  int i, result=0;
  for (i=0; i < numbits; i++)
  {
    result = (result << 1) | (input & 1);
    input >>= 1;
  }
  return result;
}  // bit reversal function


__global__ void bit_reverse (float *R, float *I, float *R_2, float *I_2, float *out_R, float *out_I, int n) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int out_index = bitrev(idx, (int)(log2f(n)));
    int check = (int)(log2f(n));
    if (check % 2 == 0)
    {
    out_R[idx] = R[out_index];
    out_I[idx] = I[out_index];
    }
    else
    {
    out_R[idx] = R_2[out_index];
    out_I[idx] = I_2[out_index];
    }
} //bit reversal kernel



__global__ void Fourier(float *R, float *I, float *R_2, float *I_2, int n, int l){
            int temp = 1<<l;
	        int idx = threadIdx.x + blockIdx.x*blockDim.x;
            int k = idx/(n/temp);
            int i = idx - k*(n/temp);
            int kmax = max(0 ,  (i - (n/(2*temp)))*temp);
            float wkreal = cos( (2*pi*((float ) kmax))/( (float) n));
            float wkimag = sin( (2*pi*((float ) kmax))/( (float) n));
            int sign = 1 - 2*(  (i/(n/(2*temp)))  % 2);
      // Switch in different level
            if (l % 2 == 0)
             {
             R_2[idx] = 0.5 * (1+sign) * (R[idx] + sign * (wkreal * R[idx + sign* (n/(2*temp))] -  wkimag * I[idx + sign* (n/(2*temp))]))  + 0.5 * (1 - sign) * (wkreal*(R[idx + sign* (n/(2*temp))]-R[idx]) - wkimag *(I[idx + sign* (n/(2*temp))]-I[idx]));

            I_2[idx] = 0.5 * (1+sign) * (I[idx] + sign * (wkreal * I[idx + sign* (n/(2*temp))]) +  wkimag * R[idx + sign* (n/(2*temp))]) + 0.5 * (1 - sign) * (wkreal * (I[idx + sign* (n/(2*temp))]-I[idx]) + wkimag *(R[idx + sign* (n/(2*temp))]-R[idx]));
             }
             else {
               R[idx] = 0.5 * (1+sign) * (R_2[idx] + sign * (wkreal * R_2[idx + sign* (n/(2*temp))] -  wkimag * I_2[idx + sign* (n/(2*temp))]))  + 0.5 * (1 - sign) * (wkreal*(R_2[idx + sign* (n/(2*temp))]-R_2[idx]) - wkimag *(I_2[idx + sign* (n/(2*temp))]-I_2[idx]));

               I[idx] = 0.5 * (1+sign) * (I_2[idx] + sign * (wkreal * I_2[idx + sign* (n/(2*temp))]) +  wkimag * R_2[idx + sign* (n/(2*temp))]) + 0.5 * (1 - sign) * (wkreal * (I_2[idx + sign* (n/(2*temp))]-I_2[idx]) + wkimag *(R_2[idx + sign* (n/(2*temp))]-R_2[idx]));
             }
} // GPU kernel for FFT


void Fourier_cpu_original (float *R, float *I, float *R_2, float *I_2, int n) {
   for (int i =0; i<n; i++){
   for (int j=0 ; j<n ; j++) {
        R_2[i] += R[j] * cos(2 * i * j * pi / n) - I[j] * sin(2 * i * j * pi / n);
        I_2[i] += R[j] * sin(2 * i * j * pi / n) + I[j] * cos(2 * i * j * pi / n);
   }
}
} //Regular FT

void Fourier_cpu (float *R, float *I, float *R_2, float *I_2, float *R_out, float *I_out, int n){
    // Fourier Calculation
   for (int l = 0; l < (int)(log(n)/log(2)); l++){
    for (int threadIdxx = 0; threadIdxx < n ; threadIdxx++){
            int temp = 1<<l;
            int k = threadIdxx/(n/temp);
            int i = threadIdxx - k*(n/temp);
            int kmax = max(0 ,  (i - (n/(2*temp)))*temp);
            float wkreal = cos( (2*pi*((float ) kmax))/( (float) n));
            float wkimag = sin( (2*pi*((float ) kmax))/( (float) n));
            int sign = 1 - 2*(  (i/(n/(2*temp)))  % 2);
            if (l % 2 == 0)
            {
            R_2[threadIdxx] = 0.5 * (1+sign) * (R[threadIdxx] + sign * (wkreal * R[threadIdxx + sign* (n/(2*temp))] -  wkimag * I[threadIdxx + sign* (n/(2*temp))]))  + 0.5 * (1 - sign) * (wkreal*(R[threadIdxx + sign* (n/(2*temp))]-R[threadIdxx]) - wkimag *(I[threadIdxx + sign* (n/(2*temp))]-I[threadIdxx]));

           I_2[threadIdxx] = 0.5 * (1+sign) * (I[threadIdxx] + sign * (wkreal * I[threadIdxx + sign* (n/(2*temp))]) +  wkimag * R[threadIdxx + sign* (n/(2*temp))]) + 0.5 * (1 - sign) * (wkreal * (I[threadIdxx + sign* (n/(2*temp))]-I[threadIdxx]) + wkimag *(R[threadIdxx + sign* (n/(2*temp))]-R[threadIdxx]));
            }
            else {
              R[threadIdxx] = 0.5 * (1+sign) * (R_2[threadIdxx] + sign * (wkreal * R_2[threadIdxx + sign* (n/(2*temp))] -  wkimag * I_2[threadIdxx + sign* (n/(2*temp))]))  + 0.5 * (1 - sign) * (wkreal*(R_2[threadIdxx + sign* (n/(2*temp))]-R_2[threadIdxx]) - wkimag *(I_2[threadIdxx + sign* (n/(2*temp))]-I_2[threadIdxx]));

              I[threadIdxx] = 0.5 * (1+sign) * (I_2[threadIdxx] + sign * (wkreal * I_2[threadIdxx + sign* (n/(2*temp))]) +  wkimag * R_2[threadIdxx + sign* (n/(2*temp))]) + 0.5 * (1 - sign) * (wkreal * (I_2[threadIdxx + sign* (n/(2*temp))]-I_2[threadIdxx]) + wkimag *(R_2[threadIdxx + sign* (n/(2*temp))]-R_2[threadIdxx]));
              
            }
        }
    }
    // Bit Reversal
    for(int j = 0; j < n; j++){
    int out_index = bitrev(j, (int)(log(n)/log(2)));
    int check = (int)(log(n)/log(2));
    if (check % 2 == 0)
    {
      R_out[j] = R[out_index];
      I_out[j] = I[out_index];
    }
    else
    {
    R_out[j] = R_2[out_index];
    I_out[j] = I_2[out_index];
    }
    }
} // CPU FFT




int main(int argc, char *argv[]){
    int num_blocks_per_grid = 1;
    int num_elements_per_block = 1024;
    int n = num_blocks_per_grid * num_elements_per_block; // n has to be 2 to some powers
    float *hA_R = (float *) malloc( n * sizeof(float) );
    float *hA_I = (float *) malloc( n * sizeof(float) );
    float *hB_R = (float *) malloc( n * sizeof(float) );
    float *hB_I = (float *) malloc( n * sizeof(float) );
    float *R = (float *) malloc( n * sizeof(float) );
    float *I = (float *) malloc( n * sizeof(float) );
    float *R_2 = (float *) malloc( n * sizeof(float) );
    float *I_2 = (float *) malloc( n * sizeof(float) );
    float *R_3 = (float *) malloc( n * sizeof(float) );
    float *I_3 = (float *) malloc( n * sizeof(float) );
    float *GPU_out_R = (float *) malloc( n * sizeof(float) );
    float *GPU_out_I = (float *) malloc( n * sizeof(float) );
    float *CPU_out_R = (float *) malloc( n * sizeof(float) );
    float *CPU_out_I = (float *) malloc( n * sizeof(float) );
    float *dA_R;
    float *dA_I;
    float *dB_R;
    float *dB_I;
    float *dout_R;
    float *dout_I;
    cudaMalloc( &dA_R, n*sizeof(float));
    cudaMalloc( &dA_I, n*sizeof(float));
    cudaMalloc( &dB_R, n*sizeof(float));
    cudaMalloc( &dB_I, n*sizeof(float));
    cudaMalloc( &dout_R, n*sizeof(float));
    cudaMalloc( &dout_I, n*sizeof(float));
    
    // set up initial value for input
    for (int i = 0; i < n; i++){
        hA_R[i] = i+1;
        R[i] = i+1;
    }
    for (int i = 0; i < n; i++){
        hA_I[i] = 1;
        I[i] = 1;
    }

    cudaMemcpy(dA_R , hA_R, n*sizeof(float), cudaMemcpyHostToDevice  );
    cudaMemcpy(dA_I , hA_I, n*sizeof(float), cudaMemcpyHostToDevice  );
    cudaMemcpy(dB_R , hB_R, n*sizeof(float), cudaMemcpyHostToDevice  );
    cudaMemcpy(dB_I , hB_I, n*sizeof(float), cudaMemcpyHostToDevice  );
    cudaMemcpy(dout_R , GPU_out_R, n*sizeof(float), cudaMemcpyHostToDevice  );
    cudaMemcpy(dout_I , GPU_out_I, n*sizeof(float), cudaMemcpyHostToDevice  );
 
    float gtime;
    float ctime;
    float c_parel_time;
    
    cstart();
    Fourier_cpu_original(R, I, R_3, I_3, n);
    cend(&ctime);

    cstart();
    Fourier_cpu(R, I, R_2, I_2, CPU_out_R, CPU_out_I, n);
    cend(&c_parel_time);
    
    
    warm_up_gpu<<<1,1,1024>>>();
    
    gstart();
    for (int l = 0; l < (int)(log(n)/log(2)); l++){
        Fourier<<< num_blocks_per_grid   ,  num_elements_per_block   >>>(dA_R, dA_I, dB_R, dB_I, n, l);
    }
    
    bit_reverse<<< num_blocks_per_grid   ,  num_elements_per_block  >>> (dA_R, dA_I, dB_R, dB_I, dout_R, dout_I, n);
    
    gend(&gtime);
    

    cudaMemcpy(GPU_out_R , dout_R, n*sizeof(float), cudaMemcpyDeviceToHost  );
    cudaMemcpy(GPU_out_I, dout_I, n*sizeof(float), cudaMemcpyDeviceToHost  );


 

   printf("CPU original time = %f, CPU Parallel time = %f, GPU time = %f \n",ctime, c_parel_time, gtime);
   
   
   // calculate the difference
   int total_count = 0;

   for (int i = 0; i < n; i++){
       int count = (GPU_out_R[i] - CPU_out_R[i])*(GPU_out_R[i] - CPU_out_R[i]) + (GPU_out_I[i] - CPU_out_I[i])*(GPU_out_I[i] - CPU_out_I[i]);
       total_count += count;
    }  

       printf("GPU result and CPU result Difference is %d \n",total_count);
 
    
       
    return 0;
}



