# Makefile to build all the available variants

BUILDS = opencvfft-st opencvfft-async fftw cufftw fftw_openmp fftw_async opencv-cufft

all: $(foreach build,$(BUILDS),build-$(build)/kcf_vot)

#CMAKE_OPTS = -DOpenCV_DIR=~/opt/opencv-2.4/share/OpenCV

CMAKE_OTPS_opencvfft-st    = -DFFT=OpenCV
CMAKE_OTPS_opencvfft-async = -DFFT=OpenCV -DASYNC=ON
CMAKE_OTPS_opencv-cufft    = -DFFT=OpenCV_cuFFT
CMAKE_OTPS_fftw            = -DFFT=fftw
CMAKE_OTPS_cufftw          = -DFFT=cuFFTW
CMAKE_OTPS_fftw_openmp     = -DFFT=fftw -DOPENMP=ON
CMAKE_OTPS_fftw_async      = -DFFT=fftw -DASYNC=ON

build-%/kcf_vot: $(shell git ls-files)
	mkdir -p $(@D)
	cd $(@D) && cmake $(CMAKE_OPTS) $(CMAKE_OTPS_$*) ..
	cmake --build $(@D)

clean:
	rm -rf $(BUILDS:%=build-%)
