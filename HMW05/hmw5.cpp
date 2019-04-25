
// for file IO
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#define PI 3.14159265359f
#include <sys/stat.h> 
#ifdef __APPLE__
#include <OpenCL/OpenCl.h>
#else
#include <CL/cl.h>
#endif

void pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data){
  fprintf(stderr, "OpenCL Error (via pfn_notify): %s\n", errinfo);
}

void oclInit(int plat, int dev,
       cl_context &context,
       cl_device_id &device,
       cl_command_queue &queue){

  /* set up CL */
  cl_int            err;
  cl_platform_id    platforms[100];
  cl_uint           platforms_n;
  cl_device_id      devices[100];
  cl_uint           devices_n ;

  /* get list of platform IDs (platform == implementation of OpenCL) */
  clGetPlatformIDs(100, platforms, &platforms_n);
  
  if( plat > platforms_n) {
    printf("ERROR: platform %d unavailable \n", plat);
    exit(-1);
  }
  
  // find all available device IDs on chosen platform (could restrict to CPU or GPU)
  cl_uint dtype = CL_DEVICE_TYPE_ALL;
  clGetDeviceIDs( platforms[plat], dtype, 100, devices, &devices_n);
  
  printf("devices_n = %d\n", devices_n);
  
  if(dev>=devices_n){
    printf("invalid device number for this platform\n");
    exit(0);
  }

  // choose user specified device
  device = devices[dev];
  
  // make compute context on device, pass in function pointer for error messaging
  context = clCreateContext((cl_context_properties *)NULL, 1, &device, &pfn_notify, (void*)NULL, &err); 

  // create command queue
  queue   = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err); // synchronized execution
}

void oclBuildKernel(const char *sourceFileName,
        const char *functionName,
        cl_context &context,
        cl_device_id &device,
        cl_kernel &kernel,
        const char *flags
        ){

  cl_int            err;

  // read in text from source file
  FILE *fh = fopen(sourceFileName, "r"); // file handle
  if (fh == 0){
    printf("Failed to open: %s\n", sourceFileName);
    throw 1;
  }

  // C function, get stats for source file (just need total size = statbuf.st_size)
  struct stat statbuf; 
  stat(sourceFileName, &statbuf); 

  // read text from source file and add terminator
  char *source = (char *) malloc(statbuf.st_size + 1); // +1 for "\0" at end
  fread(source, statbuf.st_size, 1, fh); // read in 1 string element of size "st_size" from "fh" into "source"
  source[statbuf.st_size] = '\0'; // terminates the string

  // create program from source 
  cl_program program = clCreateProgramWithSource(context,
             1, // compile 1 kernel
             (const char **) & source,
             (size_t*) NULL, // lengths = number of characters in each string. NULL = \0 terminated.
             &err); 

  if (!program){
    printf("Error: Failed to create compute program!\n");
    throw 1;
  }
    
  // compile and build program 
  err = clBuildProgram(program, 1, &device, flags, (void (*)(cl_program, void*))  NULL, NULL);

  // check for compilation errors 
  char *build_log;
  size_t ret_val_size;
  err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size); // get size of build log
  
  build_log = (char*) malloc(ret_val_size+1);
  err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, (size_t*) NULL); // read build log
  
  // to be careful, terminate the build log string with \0
  // there's no information in the reference whether the string is 0 terminated or not 
  build_log[ret_val_size] = '\0';

  // print out compilation log 
  fprintf(stderr, "%s", build_log );

  // create runnable kernel
  kernel = clCreateKernel(program, functionName, &err);
  if (! kernel || err != CL_SUCCESS){
    printf("Error: Failed to create compute kernel!\n");
    throw 1;
  }
}




int main(int argc, char **argv){

  cl_int            err;

  int plat = 0;
  int dev  = 0;

  cl_context context;
  cl_device_id device;
  cl_command_queue queue;

  cl_kernel kernel_jac;
  cl_kernel kernel_red;

  oclInit(plat, dev, context, device, queue);

  const char *sourceFileName = "jacobian.cl";
  const char *functionName = "jacobi";
  
  const char *sourceFileName1 = "reduction.cl";
  const char *functionName1 = "reduce";
  int BDIM = 32;
  char flags[BUFSIZ];
  sprintf(flags, "-DBDIM=%d", BDIM);

  oclBuildKernel(sourceFileName, functionName, context, device, kernel_jac, flags);
  oclBuildKernel(sourceFileName1, functionName1, context, device, kernel_red, flags);


  //problem implementation
  int N =128; //vector size
  int N2 = (N+2)*(N+2);
  size_t sz = N*sizeof(float);
  //int dim = 2;
  int Nt = BDIM;    
  int Ng = ((N+Nt-1))/BDIM;   //shouldn't divide by Nt???   original ((N+Nt-1)/Nt)
  Ng = BDIM*Ng;
  size_t local[3] = {Nt,Nt,1};
  size_t global[3] = {Ng,Ng,1};


  // for reduce kernel
  int Nthreads1D = BDIM; 
  //int Nblocks1D = BDIM;
  int Nblocks1D = ((N+2)*(N+2) + Nthreads1D-1)/BDIM;    //original is to divide by Nthreads1D
  Nblocks1D = Nblocks1D*BDIM;
  int halfNblocks1D = (Nblocks1D + 1)/2; 
  size_t localred[3] = {Nthreads1D,1,1};
  //  size_t globalred[3] = {halfNblocks1D,1,1};
  size_t globalred[3] = {Nblocks1D,1,1};




  float tol = 1e-6;

  float *u = (float*) calloc((N+2)*(N+2), sizeof(float));
  float *unew = (float*)calloc((N+2)*(N+2),sizeof(float));
  float *f = (float*) calloc((N+2)*(N+2), sizeof(float));
  float h = 2.0/(N+1);
  for (int i = 0; i < N+2; ++i){
    for (int j = 0; j < N+2; ++j){
      const float x = -1.0 + i*h;
      const float y = -1.0 + j*h;
      unew[i + j*(N+2)] = 1.0;
      f[i + j*(N+2)] = sin(PI*x)*sin(PI*y) * h*h;
    }
  } 
  
  cl_mem c_u = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N2*sizeof(float), u, &err);
  cl_mem c_f = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N2*sizeof(float), f, &err);
  cl_mem c_unew = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N2*sizeof(float),unew , &err);


  // storage for residual
  //float *res = (float*) calloc(halfNblocks1D, sizeof(float));
  //  cl_mem c_res = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, halfNblocks1D*sizeof(float), res, &err);
  float *res = (float*) calloc(Nblocks1D, sizeof(float));
  cl_mem c_res = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Nblocks1D*sizeof(float), res, &err);

  int iter = 0;
  float r2 = 1.;
  //arguments for jacobian kernel
  clSetKernelArg(kernel_jac, 0, sizeof(int), &N);
  clSetKernelArg(kernel_jac, 1, sizeof(cl_mem), &c_u); //c_u
  clSetKernelArg(kernel_jac, 2, sizeof(cl_mem), &c_f); //c_f
  clSetKernelArg(kernel_jac, 3, sizeof(cl_mem), &c_unew);//c_unew
  
  

  clSetKernelArg(kernel_red, 0, sizeof(int), &N2);
  clSetKernelArg(kernel_red, 1, sizeof(cl_mem), &c_u);
  clSetKernelArg(kernel_red, 2, sizeof(cl_mem), &c_unew);
  clSetKernelArg(kernel_red, 3, sizeof(cl_mem), &c_res);
  
  cl_event eventj;
  cl_event eventr;
  while (r2 > tol*tol){
   // runs jacobian kernel   
    clEnqueueNDRangeKernel(queue,kernel_jac,2,0,global,local,0,(cl_event*)NULL,&eventj);
    clWaitForEvents(1, &eventj);

 
    //runs error kernel
    clEnqueueNDRangeKernel(queue,kernel_red,1,0,globalred,localred,0,(cl_event*)NULL,&eventr);
    clWaitForEvents(1,&eventr);
    //reads back to cpu
    
      clFinish(queue);
    //    clEnqueueReadBuffer(queue, c_res, CL_TRUE, 0, halfNblocks1D*sizeof(float), res, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, c_res, CL_TRUE, 0, Nblocks1D*sizeof(float), res, 0, NULL, NULL);
    // finish block reduction on CPU

      //    printf("\n");
    r2 = 0.f;
    for (int j = 0; j < Nblocks1D; ++j){
      r2 += res[j];
    }
  //  printf("residual = %f", r2);
    ++iter;  
  }
  cl_ulong jstart, jend;
  cl_ulong rstart, rend;
  clGetEventProfilingInfo(eventj, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &jstart, NULL);
  clGetEventProfilingInfo(eventj, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &jend, NULL);

  clGetEventProfilingInfo(eventr, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &rstart, NULL);
  clGetEventProfilingInfo(eventr, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &rend, NULL);

  double jnanosecs = jend-jstart;
  double rnanosecs = rend-rstart;
  double jtime = jnanosecs/(iter);
  double rtime = rnanosecs/(iter);

  printf("jtime = %f", jtime);
  printf("\n rtime = %f", rtime);


  float error = 0.0;
  float erre = 0.0;
  for (int i = 0; i < (N+2)*(N+2); i++){
    erre = fabs(u[i]-f[i]/(h*h*2.0*PI*PI));
    //printf("%f \n", erre);
      if (erre > error){
        error = erre;
      }
  }

  
  printf("Max error: %f, r2 = %f, iterations = %d\n", error,r2,iter);

  exit(0);
}
