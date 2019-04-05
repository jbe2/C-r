#include <math.h>
#include <stdio.h>
#include <stdlib.h>



#define PI 3.14159265359

__global__ void solve(const int N, float * u, float *newu, float *f, float *res2v){
	float w = 0.5;
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if (id>N+2 && id<(N+2)*(N+2)-(N+2) && id%(N+2) !=N+1 && id%(N+2) != 0){
		const float Ru = -u[id-(N+2)] -u[id+(N+2)] - u[id-1] - u[id+1];
		const float rhs = (1./4)*(f[id]-Ru);
		const float unew = w*rhs + (1-w)*u[id];
		newu[id] = unew;
		res2v[id] = (unew-u[id])*(unew-u[id]); 
		}
	}



__global__ void res2fred(int Nblocks, int blockdim, float* res2, float* res2_small){
	int bid = blockIdx.x;
	int I =  blockDim.x*blockIdx.x+threadIdx.x;
	float extra_sum = 0;
	if (bid < Nblocks-1){
		for (int s = blockdim*(Nblocks-1)/2; s>1; s/=2){
				if (I<s){
					res2_small[I] += res2[I+s];
				} 
				__syncthreads();		 		
			}
		}
	else{
		extra_sum += res2[I];
	}
	__syncthreads();
	res2_small[Nblocks-1] = extra_sum;
}	

int main(void)
{
int N = 1000;
double tol = 1e-6;
int Nthreads = 128;
const int Nblocks = (N+Nthreads-1)/Nthreads+1;
dim3 threadsPerblock(Nthreads, 1,1);
dim3 blocks(Nblocks,1,1);
int blockdim = Nthreads;

float *f_c, *u_c, *unew_c, *res2v_c, *res2vsmall_c;
cudaMalloc(&u_c,(N+2)*(N+2)*sizeof(float));
cudaMalloc(&f_c,(N+2)*(N+2)*sizeof(float));
cudaMalloc(&unew_c,(N+2)*(N+2)*sizeof(float));
cudaMalloc(&res2v_c, (N+2)*(N+2)*sizeof(float));
cudaMalloc(&res2vsmall_c, (N+2)*(N+2)*sizeof(float));

float *u = (float*) calloc((N+2)*(N+2), sizeof(float));
float *f = (float*) calloc((N+2)*(N+2),sizeof(float));
float *unew = (float*) calloc((N+2)*(N+2), sizeof(float));
float *res2v = (float*) calloc((N+2)*(N+2), sizeof(float));
float *res2vsmall = (float*) calloc(100, sizeof(float));
float h = 2.0/(N+1);
	for (int i = 0; i<N+2; i++){
		for (int j = 0; j<N+2; j++){
			const float x = -1 + i*h;
			const float y = -1 + j*h;
			f[i+j*(N+2)] = sin(PI*x)*sin(PI*y)*h*h;
		}
	}
int iter = 0;
float res2 = 1;
float res2sum = 0;
while (res2 > tol*tol){
	cudaMemcpy(f_c, f, (N+2)*(N+2)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(u_c, u, (N+2)*(N+2)*sizeof(float), cudaMemcpyHostToDevice);
	solve <<<blocks, threadsPerblock>>> (N,u_c,unew_c, f_c, res2v_c);
	res2fred <<<blocks, threadsPerblock >>> (Nblocks, blockdim, res2v_c, res2vsmall_c);	
	cudaMemcpy(unew, unew_c, (N+2)*(N+2)*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(res2vsmall, res2vsmall_c, 100*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(res2v, res2v_c, (N+2)*(N+2)*sizeof(float), cudaMemcpyDeviceToHost);
//	for (int i = 0; i<Nblocks; i++){
//	printf("res2vsmall[%d] = %f\n",i, res2vsmall[i]);	
//	res2sum += res2vsmall[i];
//	}
	for (int i = 0; i<(N+2)*(N+2); i++){
	u[i] = unew[i];
	res2sum +=res2v[i];
	}


	res2 = res2sum;
	res2sum = 0;
//	printf("%d", iter);
	iter++;
}

float err = fabs(u[0]-f[0]/(h*h*2.0*PI*PI));
for (int i = 0; i<(N+2)*(N+2)-1; i++){
	if (err <=fabs(u[i+1]-f[i+1]/(h*h*2.0*PI*PI))){
		err = fabs(u[i+1]-f[i+1]/(h*h*2.0*PI*PI));
		}
	}
printf("iter %d", iter);
cudaFree(u_c);
cudaFree(f_c);
cudaFree(unew_c);
cudaFree(res2v_c);

free(u);
free(f);
free(unew);
free(res2v);
return 0;
}

