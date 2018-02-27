# Makefile to build all the available variants

BUILDS = opencvfft-st opencvfft-async fftw fftw-parallel fftw_openmp opencv-cufft

all: $(foreach build,$(BUILDS),build-$(build)/kcf_vot)

#CMAKE_OPTS = -DOpenCV_DIR=~/opt/opencv-2.4/share/OpenCV

CMAKE_OTPS_opencvfft-st    =
CMAKE_OTPS_opencvfft-async = -DASYNC=ON
CMAKE_OTPS_opencv-cufft    = -DOPENCV_CUFFT=ON
CMAKE_OTPS_fftw            = -DFFTW=ON
CMAKE_OTPS_fftw-parallel   = -DFFTW=ON -DFFTW_PARALLEL=ON
CMAKE_OTPS_fftw_openmp     = -D=FFTW=ON -DFFTW_OPENMP=ON


build-%/kcf_vot: $(shell git ls-files)
	mkdir -p $(@D)
	cd $(@D) && cmake $(CMAKE_OPTS) $(CMAKE_OTPS_$*) ..
	cmake --build $(@D)

clean:
	rm -rf $(BUILDS:%=build-%)
