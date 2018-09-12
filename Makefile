# Makefile to build all the available variants

BUILDS = opencvfft-st opencvfft-async opencvfft-openmp fftw fftw-async fftw-openmp fftw-big fftw-big-openmp cufftw cufftw-big cufftw-big-openmp cufft cufft-openmp cufft-big cufft-big-openmp
TESTSEQ = bag ball1 car1
TESTFLAGS = default fit128

all: $(foreach build,$(BUILDS),build-$(build)/kcf_vot)

CMAKE_OPTS += -G Ninja
#CMAKE_OPTS += -DOpenCV_DIR=~/opt/opencv-2.4/share/OpenCV

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

print-test-results = grep ^Average $(1)|sed -E -e 's|build-(.*)/kcf_vot-(.*).log:|\2;\1;|'|sort|column -t -s';'

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
	cat $$@
endef

$(foreach build,$(BUILDS),$(foreach seq,$(TESTSEQ),$(eval $(call testcase,$(build),$(seq)))))

vot2016 $(TESTSEQ:%=vot2016/%): vot2016.zip
	unzip -d vot2016 -q $^
	for i in $$(ls -d vot2016/*/); do ( echo Creating $${i}images.txt; cd $$i; ls *.jpg > images.txt ); done

.INTERMEDIATE: vot2016.zip
.SECONDARY:    vot2016.zip
vot2016.zip:
	wget http://data.votchallenge.net/vot2016/vot2016.zip
