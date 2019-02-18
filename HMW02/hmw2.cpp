# include <cstdlib>
# include <iostream>
# include <stdio.h>
# include <cmath>
# include <math.h>
# include <time.h>

# include <mpi.h>




int main(int argc, char ** argv)
{
	//clock_t  t;
	//t = clock();
	//sets up constants
	//int N = atoi(argv[1]); //N^2 is the number of interior points
	//double tol = atof(argv[2]); //test case to check iterative method

	int N = 4;  //grid number
	double tol = pow(10,-6);
	double h = 2.0/(N+1); //mesh size
	double pi = 4*atan(1);
	double a = -1;  //lower boundary of interval
	double c = 1;   //upper boundary of interval
	double w = 1.0/2.0; //weight for weighted jacobian method
	int iter = 0;  //counter to test method
	


 	MPI_Init(&argc, &argv);
 	int rank, size;
 	int root = 0;
 	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 	MPI_Comm_size(MPI_COMM_WORLD, &size);

 	MPI_Status status;

	int num = pow(N,2);
	int len;
	int rem;
	if(num%size == 0){
		len = num/size;
		rem = num/size;
	}
	else if(num%(size-1)==0){
		len = (num/size)+1;
		rem = num -(len*(size-1));
	}
	else {
		len = (num/(size-1));
		rem = num%len;
	}
	//printf("remainder %i \n", rem);
 	double *b;
 	double  *u;
 	double *u1;
 	double *ue;
 	int *num_vec;
 	int *displs;
 	if (rank == 0){
 	b = (double *) calloc (pow(N,2), sizeof(double));   //vector of u'' vals
 	u = (double *) calloc (pow(N,2), sizeof(double));    
 	u1 = (double*) calloc (pow(N,2), sizeof(double));
	ue = (double*) calloc (pow(N,2), sizeof(double));
	num_vec = (int*) calloc ((size), sizeof(int));
	displs = (int*) calloc ((size), sizeof(int));
		for (int i = 0; i < size; i++){
			if (i != size-2){
				num_vec[i] = len;
				displs[i] = i*len;
			}	
			else{
				num_vec[i] = rem;
				displs[i] = i*len;
			}
		}	
	}



 	double* bb = (double *) calloc (len, sizeof(double));   //vector of u'' vals
 	double* uu = (double *) calloc (len, sizeof(double));    
 	double* u1u1 = (double*) calloc (len, sizeof(double));
 	double* ueue = (double*) calloc (len, sizeof(double));




 	MPI_Scatterv(b,num_vec, displs, MPI_DOUBLE, bb, len, MPI_DOUBLE, 0, MPI_COMM_WORLD);  //scatters onto nodes, first n vals stay on node
 	MPI_Scatterv(u,num_vec, displs, MPI_DOUBLE, uu, len, MPI_DOUBLE, 0, MPI_COMM_WORLD);  //scatters onto nodes, first n vals stay on node
 	MPI_Scatterv(u1,num_vec, displs, MPI_DOUBLE, u1u1, len, MPI_DOUBLE, 0, MPI_COMM_WORLD);  //scatters onto nodes, first n vals stay on node
 	MPI_Scatterv(ue,num_vec, displs, MPI_DOUBLE, ueue, len, MPI_DOUBLE, 0, MPI_COMM_WORLD);  //scatters onto nodes, first n vals stay on node


 	//INITIALIZING VALUES FOR VECTORS ACROSS RANKS
	MPI_Barrier(MPI_COMM_WORLD);
	for (int i = 0; i<len; i++){
		u1u1[i] = 0;
		ueue[i] = 0;
		int pos = (rank)*len + i;
		int x_pos = (pos%(N));
		int y_pos = pos/(N);
		double x = a + x_pos*h;
		double y = a + y_pos*h;
		if(x_pos ==0 || x_pos == N-1 || y_pos == 0 || y_pos == N-1){
			bb[pos] = 0;
			uu[pos]= 0;
		}
		else{
			bb[i] = sin(x*pi)*sin(y*pi);
			uu[i] = 1;
		}

	}
	MPI_Barrier(MPI_COMM_WORLD);


	double E = tol+1;
	while(E>=tol){
		//Identifies whihc elements to send to adjascent rank
		if (rank == 0){
			int count = 0;
			int x_pos;
			int y_pos;
			int x_end = (len-1)%(N);
			int y_end = (len-1)/N;
			double u_bef_s = uu[len-1];
			double u_aft_r;
			double *send_up = (double*)calloc(N,sizeof(double));
			double *recv_down = (double*)calloc(N,sizeof(double));
			for (int i = 0; i<len; i++){

				x_pos = (len-1)%N;
				y_pos = (len-1)/N;
				if (x_pos == 0 || x_pos == N-1 || y_pos == 0 || y_pos == N-1){
					uu[i] = 0;
				}
				if(len%N == 0){
					if(len<N){
						count = len;
						for (int i = 0; i<len; i++){
							send_up[i] = uu[i];
						}
					}
					else{
					count = N;
						for (int i = 0; i<N; i++){
							send_up[i] = uu[len-N+i];
						}
					}	
				}
				else{
					if(y_pos == y_end){
						count++;
					}
					else if (x_pos>x_end && y_pos == y_end-1){
						count++;
					}
				}
			}
			for (int i = 0; i<count; i++){
				send_up[i] = u[len-count+i];
			}
			int sendmin;
			int recvmin;
			if (len<N){
				sendmin = len;
			}
			else{
				sendmin = N;
			}
			if(rem<N){
				recvmin = rem;
			}
			else{
				recvmin = N;
			}
			MPI_Send(&u_bef_s, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
			MPI_Send(&send_up[0], sendmin, MPI_DOUBLE, 1, sendmin, MPI_COMM_WORLD);
			MPI_Recv(&u_aft_r, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);
			MPI_Recv(&recv_down[0], recvmin, MPI_DOUBLE, 1, recvmin, MPI_COMM_WORLD, &status);
			MPI_Barrier;

			//calculates the next iterate
			double uu_lpos;
			double uu_rpos;
			double uu_bpos;
			double uu_upos;
			double sum = 0;
			for (int i = 0; i<len; i++){
				double uu_pos = uu[i];
				if (i != 0){
					 uu_lpos = uu[i-1];
					if (i<len-1){
						 uu_rpos = uu[i+1];
						
						if (i+N+2<len){
							 uu_upos = uu[i+N+2];
							
							if(i-(N+2) > 0){
								 uu_bpos = uu[i-N-2];
							}
							else{
								 uu_bpos = 0;
							}
						}
						else{
							 uu_upos = recv_down[i];
						}
					}
					else{
						 uu_rpos = u_aft_r;
						}
				}
				else{
					 uu_lpos = 0;
				}
				 sum +=	(1/pow(h,2))*(uu_lpos + uu_rpos + uu_bpos + uu_upos);
				u1u1[i] = (w*0.25*(pow(h,2)))*(bb[i] - sum) + (1-w)*uu[i];   
				ue[i] = fabs((1/(h*h))*(-1*(uu_lpos+uu_rpos+uu_bpos+uu_upos) + 4*uu_pos)-bb[i]);   
			}	
		}
		else if (rank == size-1){
			int count = 0;
			int pos;
			int x_pos;
			int y_pos;
			int x_start = (len*(size-1))%N;
			int y_start = (len*(size-1))/N;
			int x_end = N-1;
			int y_end = N-1;
			double u_aft_s = uu[0];
			double u_bef_r;	
			double *send_down = (double*) calloc(N,sizeof(double));
			double *recv_up =  (double*) calloc(N,sizeof(double));
			for (int i = 0; i<rem; i++){
				int pos = (len*(size-1))+i;
				x_pos = pos%N;
				y_pos = pos/N;
				if (x_pos == 0 || x_pos == N-1 || y_pos == 0 || y_pos == N-1){
					uu[i] = 0;
				}
				if (rem == N){
					send_down[i] = uu[i];
				}
				else if(rem<N){
					for (int i = 0; i<rem; i++){
						send_down[i] = uu[i];
					}
				}
				else{
					for (int i = 0; i<N; i++){
						send_down[i] = uu[i];
					}
				}

			}
			int sendmin;
			int recvmin;
			if (rem <=N){
				sendmin = rem;
			}
			else{
				sendmin = N;
			}
			if (N<=len){
				recvmin = N;
			}
			else{
			 recvmin = len;
			 MPI_Send(&u_aft_s, 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD);
			 MPI_Send(&send_down[0], sendmin, MPI_DOUBLE, rank-1, sendmin, MPI_COMM_WORLD);
			 MPI_Recv(&u_bef_r, 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &status);
			 MPI_Recv(&recv_up[0], recvmin, MPI_DOUBLE, rank-1, recvmin, MPI_COMM_WORLD, &status);
			 MPI_Barrier; 
			 

			 //calculates next iterate
			 double uu_lpos;
			 double uu_rpos;
			for (int i = 0; i<len; i++){
				if (i != 0){
					  uu_lpos = uu[i-1];
					if(i < rem-1){
						  uu_rpos = uu[i+1];
					} 
					else{
						  uu_rpos = 0;
					}
				}
				else{
					  uu_lpos = u_bef_r;
				}
				double uu_upos = 0;
				double uu_bpos = recv_up[i];  
				double sum = (1/pow(h,2))*(uu_upos + uu_bpos + uu_lpos + uu_rpos);
				u1u1[i] = (w*0.25*(pow(h,2)))*(bb[i] - sum) + (1-w)*uu[i];   
				ueue[i] = fabs((1/(h*h))*(-1*(uu_lpos+uu_rpos+uu_bpos+uu_upos) + 4*uu_upos)-bb[i]);   
				}
			}
		}
		else{
			//finds which elements ot share with adjascent nodes
			int min;
			if (N<=len){
				min = N;
			}
			else{
				min = len;
			}
			int count = 0;
			int pos;
			int x_pos;
			int y_pos;
			int x_start = ((rank)*len)%N;
			int y_start = ((rank-1)*len)/N;
			int x_end = (((rank-1))*len +len-1)%N;
			int y_end = (((rank-1))*len+len-1)/N;
			double u_aft_s = uu[0];
			double u_bef_s = uu[len-1];
			double u_aft_r;
			double u_bef_r;
			double *(send_down) = (double*) calloc(min,sizeof(double));
			double *(rec_down) =  (double*)  calloc(min,sizeof(double));
			double *(send_up);
			double *(rec_up);
			if (rank <size-2){
				double *(send_up) = (double*) calloc(min,sizeof(double));
				double *(rec_up) = (double*) calloc(min,sizeof(double));
			}
			else{
				double *(send_up) = (double*)calloc(rem,sizeof(double));
				double *(rec_up) = (double*) calloc(rem,sizeof(double));
			}
			 for (int i = 0; i<len; i++){
			//doesn't fill vectors correctly in this forloop

				pos = (rank)*len +i;
				x_pos =pos%N;
				y_pos = pos/N;

				if (x_pos ==0 || x_pos == N-1 || y_pos == 0 || y_pos == N-1){
					uu[i] = 0;
				}
				if (x_pos >x_start){
					if(y_pos == y_start){
						send_down[i] = uu[i];
					}
					
				}
				else if (x_pos < x_start){
					if (y_pos == y_start+1){
						send_down[i] = uu[i];
					}
				}

				if (x_pos>x_end){
				 	if (y_pos == y_end-1){
						send_up[i] = uu[i];
					}
				
				}
				else if(x_pos<x_end){
					if(y_pos == y_end){
					send_up[i] == uu[i];
					}
				}
			}

			MPI_Send(&u_aft_s, 1, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD);
			MPI_Recv(&u_aft_r, 1, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &status);
			MPI_Send(&u_bef_s, 1, MPI_DOUBLE, rank+1, 2, MPI_COMM_WORLD);
			MPI_Recv(&u_bef_r, 1, MPI_DOUBLE, rank-1, 2, MPI_COMM_WORLD, &status);
			if (rank =size-2){
				MPI_Send(&send_up[0], rem, MPI_DOUBLE, rank+1, rem, MPI_COMM_WORLD);
				MPI_Recv(&rec_up[0], rem, MPI_DOUBLE, rank-1, rem, MPI_COMM_WORLD, &status);
			}
			else{
				MPI_Send(&send_up[0], min, MPI_DOUBLE, rank+1, min, MPI_COMM_WORLD);
				MPI_Recv(&rec_up[0], min, MPI_DOUBLE, rank-1, min, MPI_COMM_WORLD, &status);
			}
			MPI_Send(&send_down[0], min, MPI_DOUBLE, rank-1, min, MPI_COMM_WORLD);
			MPI_Recv(&send_down[0], min, MPI_DOUBLE, rank+1, min, MPI_COMM_WORLD, &status);
			MPI_Barrier;


			//calculing the next iterate
			double uu_lpos;
			double uu_rpos;
			double ubef_r;
			double *senda = (double*)calloc(rem, sizeof(double));
			double *senda_r = (double*)calloc(rem, sizeof(double));
			for (int i = 0; i<len; i++){
				int tag = 0;
				int pos = rank*len + i;
 				int x_pos = pos%(N+2);
				int y_pos = pos/(N+2);
				double senda[len] = {};
				if (x_pos == 0 || x_pos == N+1 || y_pos == 0 || y_pos == N+1){   //check if its on the boundary
					uu[pos] = 0;
				}
	 			senda[i] = uu[i];
	 		}
	 		MPI_Send(senda,rem,MPI_DOUBLE,rank-1, 0,MPI_COMM_WORLD);    //send down a rank
	 		MPI_Recv(&ubef_r, 1, MPI_DOUBLE, N-1, 0, MPI_COMM_WORLD, &status);
	 		MPI_Recv(&senda_r, rem, MPI_DOUBLE,N-1,0,MPI_COMM_WORLD, &status);
	 		MPI_Barrier(MPI_COMM_WORLD);
	 		for (int i = 0; i<len; i++){
				if (i != 0){
					 uu_lpos = uu[i-1];
					if(i < rem-1){
						 uu_rpos = uu[i+1];
					} 
					else{
						 uu_rpos = 0;
					}
				}
				else{
					 uu_lpos = ubef_r;
				}
				double uu_upos = 0;
				double uu_bpos = senda_r[i];   //make sure you sent a senda
				double sum = (1/pow(h,2))*(uu_upos + uu_bpos + uu_lpos + uu_rpos);
				u1u1[i] = (w*0.25*(pow(h,2)))*(bb[i] - sum) + (1-w)*uu[i];   
				ueue[i] = fabs((1/(h*h))*(-1*(uu_lpos+uu_rpos+uu_bpos+uu_upos) + 4*uu_upos)-bb[i]);   
			}
		}	
		MPI_Barrier(MPI_COMM_WORLD);
		//checks if error is above tolerance
		MPI_Gatherv(u1u1, len, MPI_DOUBLE, u1, num_vec, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Gatherv(ueue, len, MPI_DOUBLE, ue, num_vec, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		E = ue[0];
		for (int i = 0; i<(N+2)*(N+2); i++){
			if (ue[i] >= E){
				E = ue[i];
			}
			u[i] = u1[i];
		}
		MPI_Scatter(u, len, MPI_DOUBLE, uu, len, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}

	MPI_Finalize();	
}