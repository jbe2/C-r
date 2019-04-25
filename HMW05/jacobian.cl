__kernel void jacobi(int N, __global float * u, __global float *f, __global float *unew){
  
  const int i = get_local_id(0) + get_local_size(0)*get_group_id(0) + 1; 
  const int j = get_local_id(1) + get_local_size(1)*get_group_id(1) + 1;

  if (i < N+1 && j < N+1){
    const int Np = (N+2);
    const int id = i + j*(N+2);
    const float ru = -u[id-Np]-u[id+Np]-u[id-1]-u[id+1];
    const float newu = .25 * (f[id] - ru);
    unew[id] = newu;
  }
}
