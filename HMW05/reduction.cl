__kernel void reduce(int N2, __global  float * u, __global float * unew, __global float *res){

  __local volatile float shared_vec[BDIM];

  const int tid = get_local_id(0);
  //  const int i = tid + get_group_id(0)*(2*get_local_size(0));
  const int i = tid + get_group_id(0)*(get_local_size(0));

  shared_vec[tid] = 0;
  if (i < N2){
    const float unew1 = unew[i];
    const float diff1 = unew1 - u[i];
    //    float unew2 = 0;
    //    float diff2 = 0;
    //    if (i + get_local_size(0)<N2){
    //      unew2 = unew[i + get_local_size(0)];
    //      diff2 = unew2 - u[i + get_local_size(0)];
    //    }

    //shared_vec[tid] = diff1*diff1 + diff2*diff2; 
    shared_vec[tid] = diff1*diff1;

    // update u
    u[i] = unew1;
    //    if (i+get_local_size(0)<N2){
    //      u[i + get_local_size(0)] = unew2;
    //    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
 
  //  for (unsigned int s = get_local_size(0)/2; s > 0; s /= 2){
  //  if (tid < s){
  //      shared_vec[tid] += shared_vec[tid+s]; // no wasted threads on first iteration
  //   }
  //   barrier(CLK_LOCAL_MEM_FENCE);
  // }   
  //
  //  if (tid==0){
  //   res[get_group_id(0)] = shared_vec[i];
  // }

  for (unsigned int s = 1; s<get_local_size(0); s*=2){
    int index = 2*s*tid;
    if (index <get_local_size(0)){
      shared_vec[index] +=shared_vec[index+s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0){
    res[get_group_id(0)] = shared_vec[0];

  }
}
