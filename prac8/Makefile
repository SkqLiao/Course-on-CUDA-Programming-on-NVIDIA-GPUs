
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_70 --ptxas-options=-v --use_fast_math

scan:	scan.cu Makefile
	nvcc scan.cu -o scan $(INC) $(NVCCFLAGS) $(LIB)

scan2:	scan2.cu Makefile
	nvcc scan2.cu -o scan2 $(INC) $(NVCCFLAGS) $(LIB)

scan3:	scan3.cu Makefile
	nvcc scan3.cu -o scan3 $(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm scan scan2 scan3

