# Compiler
CC = g++
NVCC = nvcc

# Compiler flags
CFLAGS = -Wall
CUDAFLAGS = -G

# Source files for different targets
TEST_SRCS = utils/utilsTest.cc utils/vectorUtils.cc utils/matrixUtils.cc
QR_SRCS = qrDecomp.cc utils/vectorUtils.cc matrixUtils.cc
CALIB_SRCS = calib.cc calibFile.cc utils/vectorUtils.cc utils/matrixUtils.cc
MATCHING_SRCS = matching.cc utils/imageUtils.cc utils/timer.cc makeDepthMap.cc
MATCHING_P_SRCS = matchingP.cu utils/imageUtils.cc utils/timer.cc makeDepthMap.cc

# Object files (replace .cc with .o)
TEST_OBJS = $(TEST_SRCS:.cc=.o)
QR_OBJS = $(QR_SRCS:.cc=.o)
CALIB_OBJS = $(CALIB_SRCS:.cc=.o)
MATCHING_OBJS = $(MATCHING_SRCS:.cc=.o)
MATCHING_P_OBJS = $(MATCHING_P_SRCS:.cu=.o)

# Output executables
TEST_TARGET = tests
QR_TARGET = qr
CALIB_TARGET = calib
MATCHING_TARGET = matching
MATCHING_P_TARGET = matchingP

# Generic rule to build the final executable
$(TEST_TARGET): $(TEST_OBJS)
	$(CC) $(CFLAGS) -o $@ $(TEST_OBJS)

$(QR_TARGET): $(QR_OBJS)
	$(CC) $(CFLAGS) -o $@ $(QR_OBJS)

$(CALIB_TARGET): $(CALIB_OBJS)
	$(CC) $(CFLAGS) -o $@ $(CALIB_OBJS)

$(MATCHING_TARGET): $(MATCHING_OBJS)
	$(CC) $(CFLAGS) -o $@ $(MATCHING_OBJS) -ljpeg

$(MATCHING_P_TARGET): $(MATCHING_P_SRCS)
	$(NVCC) $(CUDAFLAGS) -o $@ $(MATCHING_P_SRCS) -ljpeg

# Rule to build object files
%.o: %.cc
	$(CC) $(CFLAGS) -c $< -o $@
%.o: %.cu
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(MATCHING_OBJS) $(TEST_OBJS) $(QR_OBJS) $(CALIB_OBJS) $(MATCHING_TARGET) $(TEST_TARGET) $(QR_TARGET) $(CALIB_TARGET) $(MATCHING_P_TARGET)

# Generic run rule
run: $(TARGET)
	./$(TARGET)

# Specific run targets
run_tests: TARGET=$(TEST_TARGET)
run_tests: run

run_qr: TARGET=$(QR_TARGET)
run_qr: run

run_calib: TARGET=$(CALIB_TARGET)
run_calib: run

run_matching: TARGET=$(MATCHING_TARGET)
run_matching: run

run_matchingP: TARGET=$(MATCHING_P_TARGET)
run_matchingP: run


# ==== To run the tests ====
# make tests		// build
# make run_tests	// run

# ==== To run the calibration ====
# make calib		// build
# make run_calib	// run

# ==== To clean any build/object files ====
# make clean
