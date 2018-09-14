# Makefile to build all the available variants

BUILDS = opencvfft-st opencvfft-async opencvfft-openmp fftw fftw-async fftw-openmp fftw-big fftw-big-openmp cufftw cufftw-big cufftw-big-openmp cufft cufft-openmp cufft-big cufft-big-openmp
TESTSEQ = bag ball1 car1 book
TESTFLAGS = default fit128

all: $(foreach build,$(BUILDS),build-$(build)/kcf_vot)

CMAKE_OPTS += -G Ninja

## Useful setting - uncomment and modify as needed
# CMAKE_OPTS += -DOpenCV_DIR=~/opt/opencv-2.4/share/OpenCV
# CMAKE_OPTS += -DCUDA_VERBOSE_BUILD=ON -DCUDA_NVCC_FLAGS="--verbose;--save-temps"
# export CC=gcc-5
# export CXX=g++-5
# export CUDA_BIN_PATH=/usr/local/cuda-9.0

CMAKE_OTPS_opencvfft-st      = -DFFT=OpenCV
CMAKE_OTPS_opencvfft-async   = -DFFT=OpenCV -DASYNC=ON
CMAKE_OTPS_opencvfft-openmp  = -DFFT=OpenCV -DOPENMP=ON
CMAKE_OTPS_fftw              = -DFFT=fftw
CMAKE_OTPS_fftw-openmp       = -DFFT=fftw -DOPENMP=ON
CMAKE_OTPS_fftw-async        = -DFFT=fftw -DASYNC=ON
CMAKE_OTPS_fftw-big          = -DFFT=fftw -DBIG_BATCH=ON
CMAKE_OTPS_fftw-big-openmp   = -DFFT=fftw -DBIG_BATCH=ON -DOPENMP=ON
CMAKE_OTPS_cufftw            = -DFFT=cuFFTW
CMAKE_OTPS_cufftw-big        = -DFFT=cuFFTW -DBIG_BATCH=ON
CMAKE_OTPS_cufftw-big-openmp = -DFFT=cuFFTW -DBIG_BATCH=ON -DOPENMP=ON
CMAKE_OTPS_cufft             = -DFFT=cuFFT
CMAKE_OTPS_cufft-openmp	     = -DFFT=cuFFT -DOPENMP=ON
CMAKE_OTPS_cufft-big         = -DFFT=cuFFT -DBIG_BATCH=ON
CMAKE_OTPS_cufft-big-openmp  = -DFFT=cuFFT -DBIG_BATCH=ON -DOPENMP=ON

.SECONDARY: $(BUILDS:%=build-%/build.ninja)

build-%/build.ninja:
	@echo '############################################################'
	mkdir -p $(@D)
	cd $(@D) && cmake $(CMAKE_OPTS) $(CMAKE_OTPS_$*) ..

.PHONY: FORCE
build-%/kcf_vot: build-%/build.ninja $(shell git ls-files)
	@echo '############################################################'
	cmake --build $(@D)

$(BUILDS): %: build-%/kcf_vot

clean:
	rm -rf $(BUILDS:%=build-%)

##########################
### Tests
##########################

print-test-results = grep ^Average $(1)|sed -E -e "s|build-(.*)/kcf_vot-(.*).log:|\2;\1;|"|sort|column -t -s";"

test: $(BUILDS:%=test-%)
	@echo; echo "Summary test results:"
	@$(call print-test-results,$(foreach build,$(BUILDS),\
				   $(foreach seq,$(TESTSEQ),\
				   $(foreach f,$(TESTFLAGS),build-$(build)/kcf_vot-$(seq)-$(f).log))))

$(BUILDS:%=test-%): test-%:
	@$(call print-test-results,$(foreach seq,$(TESTSEQ),\
				   $(foreach f,$(TESTFLAGS),build-$*/kcf_vot-$(seq)-$(f).log)))

# Usage: testcase <build> <seq>
define testcase
test-$(1): test-$(1)-$(2)
test-$(1)-$(2): $(foreach f,$(TESTFLAGS),build-$(1)/kcf_vot-$(2)-$(f).log)
$(foreach f,$(TESTFLAGS),build-$(1)/kcf_vot-$(2)-$(f).log): build-$(1)/kcf_vot $$(filter-out %/output.txt,$$(wildcard vot2016/$(2)/*)) | vot2016/$(2)
	$$< $$(if $$(@:%fit128.log=),,--fit=128) vot2016/$(2) > $$@
	$(if $(TRAVIS),,cat $$@)
endef

$(foreach build,$(BUILDS),$(foreach seq,$(TESTSEQ),$(eval $(call testcase,$(build),$(seq)))))

vot2016 $(TESTSEQ:%=vot2016/%): vot2016.zip
	unzip -d vot2016 -q $^
	for i in $$(ls -d vot2016/*/); do ( echo Creating $${i}images.txt; cd $$i; ls *.jpg > images.txt ); done

.INTERMEDIATE: vot2016.zip
.SECONDARY:    vot2016.zip
vot2016.zip:
	wget http://data.votchallenge.net/vot2016/vot2016.zip

###################
# Ninja generator #
###################

# Building all $(BUILDS) with make is slow, even when run with in
# parallel (make -j). The target below generates build.ninja file that
# compiles all variants in the same ways as this makefile, but faster.
# The down side is that the build needs about 10 GB of memory.

ninja: build.ninja
	ninja

define nl


endef

define echo
echo $(1) '$(subst $(nl),\n,$(subst \,\\,$(2)))';
endef

# Ninja generator - to have faster parallel builds and tests
.PHONY: build.ninja
build.ninja:
	@$(call echo,>$@,$(ninja-rule))
	@$(foreach build,$(BUILDS),\
		$(call echo,>>$@,$(call ninja-build,$(build),$(CMAKE_OTPS_$(build)))))
	@$(foreach build,$(BUILDS),$(foreach seq,$(TESTSEQ),$(foreach f,$(TESTFLAGS),\
		$(call echo,>>$@,$(call ninja-testcase,$(build),$(seq),$(f)))$(nl))))
	@$(call echo,>>$@,build test: print_results $(foreach build,$(BUILDS),$(foreach seq,$(TESTSEQ),$(foreach f,$(TESTFLAGS),$(call ninja-test,$(build),$(seq),$(f))))))
	@$(foreach build,$(BUILDS),$(call echo,>>$@,build test-$(build): print_results $(foreach seq,$(TESTSEQ),$(foreach f,$(TESTFLAGS),$(call ninja-test,$(build),$(seq),$(f))))))
	@$(foreach seq,$(TESTSEQ),$(call echo,>>$@,build test-$(seq): print_results $(foreach build,$(BUILDS),$(foreach f,$(TESTFLAGS),$(call ninja-test,$(build),$(seq),$(f))))))
	@$(foreach seq,$(TESTSEQ),$(call echo,>>$@,build vot2016/$(seq): make))


define ninja-rule
rule cmake
  command = cd $$$$(dirname $$out) && cmake $(CMAKE_OPTS) $$opts ..
  description = CMake $$out
rule ninja
  # Absolute path in -C allows Emacs to properly jump to error message locations
  command = ninja -C `realpath $$$$(dirname $$out)`
  description = Ninja $$out
rule test_seq
  command = build-$$build/kcf_vot $$flags $$seq > $$out
rule print_results
  description = Print results
  command = $(call print-test-results,$$in)
rule make
  command = make $$out
endef

define ninja-build
build build-$(1)/build.ninja: cmake
  opts = $(2)
build build-$(1)/kcf_vot: ninja build-$(1)/build.ninja build.ninja
default build-$(1)/kcf_vot
endef

ninja-test = build-$(1)/kcf_vot-$(2)-$(3).log

# Usage: ninja-testcase <build> <seq> <flags>
define ninja-testcase
build build-$(1)/kcf_vot-$(2)-$(3).log: test_seq build-$(1)/kcf_vot $(filter-out %/output.txt,$(wildcard vot2016/$(2)/*)) || vot2016/$(2)
  build = $(1)
  seq = vot2016/$(2)
  flags = $(if $(3:fit128=),,--fit=128)
endef
