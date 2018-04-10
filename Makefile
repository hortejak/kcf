# Makefile to build all the available variants

BUILDS = opencvfft-st opencvfft-async fftw fftw_openmp fftw_async fftw_big fftw_big_openmp cufftw cufftw_big cufftw_big_openmp cufft cufft_big cufft_big_openmp

all: $(foreach build,$(BUILDS),build-$(build)/kcf_vot)

CMAKE_OPTS += -G Ninja
#CMAKE_OPTS += = -DOpenCV_DIR=

CMAKE_OTPS_opencvfft-st      = -DFFT=OpenCV
CMAKE_OTPS_opencvfft-async   = -DFFT=OpenCV -DASYNC=ON
#CMAKE_OTPS_opencv-cufft    = -DFFT=OpenCV_cuFFT
CMAKE_OTPS_fftw              = -DFFT=fftw
CMAKE_OTPS_fftw_openmp       = -DFFT=fftw -DOPENMP=ON
CMAKE_OTPS_fftw_async        = -DFFT=fftw -DASYNC=ON
CMAKE_OTPS_fftw_big          = -DFFT=fftw -DBIG_BATCH=ON
CMAKE_OTPS_fftw_big_openmp   = -DFFT=fftw -DBIG_BATCH=ON -DOPENMP=ON
CMAKE_OTPS_cufftw            = -DFFT=cuFFTW
CMAKE_OTPS_cufftw_big        = -DFFT=cuFFTW -DBIG_BATCH=ON
CMAKE_OTPS_cufftw_big_openmp = -DFFT=cuFFTW -DBIG_BATCH=ON -DOPENMP=ON
CMAKE_OTPS_cufft             = -DFFT=cuFFT
CMAKE_OTPS_cufft_big         = -DFFT=cuFFT -DBIG_BATCH=ON
CMAKE_OTPS_cufft_big_openmp  = -DFFT=cuFFT -DBIG_BATCH=ON -DOPENMP=ON

build-%/kcf_vot: $(shell git ls-files)
	mkdir -p $(@D)
	cd $(@D) && cmake $(CMAKE_OPTS) $(CMAKE_OTPS_$*) ..
	cmake --build $(@D)

$(BUILDS): %: build-%/kcf_vot

clean:
	rm -rf $(BUILDS:%=build-%)
