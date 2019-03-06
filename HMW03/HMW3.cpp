# include <cstdlib>
# include <iostream>
# include <stdio.h>
# include <cmath>
# include <math.h>
# include <time.h>
# include <omp.h>




int main()
{
	double start;
	double end;
	start = omp_get_wtime();
	//sets up constants
	int N = 10; //N^2 is the number of interior points
	double tol = 1e-6; //test case to check iterative method
	double size = 4;
	double h = 2.0/(N+1); //mesh size
	double pi = 4*atan(1);
	double a = -1;  //lower boundary of interval
	double c = 1;   //upper boundary of interval
	double w = 1.0/2.0; //weight for weighted jacobian method
	int iter = 0;  //counter to test method
	double* b = (double *) calloc ((N+2)*(N+2), sizeof(double));   //vector of u'' vals
	double* u = (double *) calloc ((N+2)*(N+2), sizeof(double));    
	double* u1 = (double*) calloc ((N+2)*(N+2), sizeof(double));
	double* ue = (double*) calloc ((N+2)*(N+2), sizeof(double));


	omp_set_num_threads(size);

	# pragma omp collapse(2)
	{
	//establish b vector and  u vector
	for (int j = 0; j < N+2; j++){    
		for (int i = 0; i<N+2; i++){   
			int pos = i + j*(N+2); //conversion from cartesian coordinates to vector index 

			//initializing vectors
			u1[pos] = 0; 
			ue[pos] = 0;

			double x = a + i*h;  //setting x position N   
			double y = a + j*h;  //setting y position N     

			
			//checking if the elements are on the boundary -- force to be zero from homogeneous dirichlet conditions
			if (i == 0 || j == 0 || i == N+1 || j == N+1) {
				b[pos] = 0;
				u[pos] = 0;

			}
			//evaluating interior points as f, or the first iteration of u
			else {
				b[pos] = sin(x*pi)*sin(y*pi);   
				u[pos] = 1;
			}

		}
	}
	}
	
	double E = tol+1; //setting the error above the tolerance    
	while (E >=tol){

		# pragma omp collapse(2)
		{
		for (int j = 1; j<N+1; j++){
			for (int i = 1; i<N+1; i++){
				//index the positions of the 5 point stencil
				int pos = i + j*(N+2);     
				int lpos = i-1 + j*(N+2);  
				int rpos = i+1 + j*(N+2);  
				int bpos = i + (j-1)*(N+2);  
				int upos = i + (j+1)*(N+2);  

				double sum = 0;

				//create the sum to subtract from in the weighted jacobian method
				sum += (-1/pow(h, 2))*(u[lpos] + u[rpos] + u[bpos] + u[upos]);        

				//calculate the next iteration with the weighted jacobian method
				u1[pos] = (w*0.25*(pow(h,2)))*(b[pos] - sum) + (1-w)*u[pos];    
				
			}
		}	
		}
		 
		# pragma omp collapse(2)
		{
		for (int j = 1; j<N+1; j++){
			for (int i = 1; i<N+1; i++){
				//indexing the 5 point stencil         
				int pos = i + j*(N+2);    // (i,j)
				int lpos = i-1 + j*(N+2); //(i-1,j)   (left)
				int rpos = i+1 + j*(N+2); //(i+1,j)   (right)
				int bpos = i + (j-1)*(N+2);  //(i,j-1)  (below)
				int upos = i + (j+1)*(N+2);  //(i,j+1)  (upper)
				//calculating the  error
				ue[pos] = fabs((1/(h*h))*(-1*(u[lpos]+u[rpos]+u[bpos]+u[upos]) + 4*u[pos])-b[pos]);   
				
			}
		}
		}
		# pragma omp critical
		E = ue[0];
		//finding the l-infinity norm for error vector
		for (int i = 1; i<pow(N+2,2); i++){
			if (ue[i] > E){
				E = ue[i];
			}
			u[i] = u1[i];
		}

		iter++;
	}
	end = omp_get_wtime();
	//for (int i = 0; i<(N+2)*(N+2); i++){
//		printf("%f\n", u1[i]);
//	}
	printf("itrations required = %i ", iter);
	printf("total time = %f", end-start);
	printf("\n");

	

	return 0;
}
