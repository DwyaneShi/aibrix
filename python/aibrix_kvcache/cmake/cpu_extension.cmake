include(FetchContent)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOSX_FOUND TRUE)
endif()


#
# Define environment variables for special configurations
#
if(DEFINED ENV{AIBRIX_CPU_AVX512BF16})
    set(ENABLE_AVX512BF16 ON)
endif()

include_directories("${CMAKE_SOURCE_DIR}/csrc")


set (ENABLE_NUMA TRUE)

#
# Check the compile flags
#

if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    list(APPEND AIBRIX_CPU_FLAGS
        "-mf16c"
    )
endif()

if(MACOSX_FOUND)
    list(APPEND AIBRIX_CPU_FLAGS
        "-DAIBRIX_CPU_EXTENSION")
else()
    list(APPEND AIBRIX_CPU_FLAGS
        "-fopenmp"
        "-DAIBRIX_CPU_EXTENSION")
endif()

if (NOT MACOSX_FOUND)
    execute_process(COMMAND cat /proc/cpuinfo
                    RESULT_VARIABLE CPUINFO_RET
                    OUTPUT_VARIABLE CPUINFO)
    if (NOT CPUINFO_RET EQUAL 0)
        message(FATAL_ERROR "Failed to check CPU features via /proc/cpuinfo")
    endif()
endif()


function (find_isa CPUINFO TARGET OUT)
    string(FIND ${CPUINFO} ${TARGET} ISA_FOUND)
    if(NOT ISA_FOUND EQUAL -1)
        set(${OUT} ON PARENT_SCOPE)
    else()
        set(${OUT} OFF PARENT_SCOPE)
    endif()
endfunction()

function (is_avx512_disabled OUT)
    set(DISABLE_AVX512 $ENV{AIBRIX_CPU_DISABLE_AVX512})
    if(DISABLE_AVX512 AND DISABLE_AVX512 STREQUAL "true")
        set(${OUT} ON PARENT_SCOPE)
    else()
        set(${OUT} OFF PARENT_SCOPE)
    endif()
endfunction()

is_avx512_disabled(AVX512_DISABLED)

if (MACOSX_FOUND AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    set(APPLE_SILICON_FOUND TRUE)
else()
    find_isa(${CPUINFO} "avx2" AVX2_FOUND)
    find_isa(${CPUINFO} "avx512f" AVX512_FOUND)
    find_isa(${CPUINFO} "Power11" POWER11_FOUND)
    find_isa(${CPUINFO} "POWER10" POWER10_FOUND)
    find_isa(${CPUINFO} "POWER9" POWER9_FOUND)
    find_isa(${CPUINFO} "asimd" ASIMD_FOUND) # Check for ARM NEON support
    find_isa(${CPUINFO} "bf16" ARM_BF16_FOUND) # Check for ARM BF16 support
    find_isa(${CPUINFO} "S390" S390_FOUND)
endif()


if (AVX512_FOUND AND NOT AVX512_DISABLED)
    list(APPEND AIBRIX_CPU_FLAGS
        "-mavx512f"
        "-mavx512vl"
        "-mavx512bw"
        "-mavx512dq")

    find_isa(${CPUINFO} "avx512_bf16" AVX512BF16_FOUND)
    if (AVX512BF16_FOUND OR ENABLE_AVX512BF16)
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
            CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.3)
            list(APPEND AIBRIX_CPU_FLAGS "-mavx512bf16")
        else()
            message(WARNING "Disable AVX512-BF16 ISA support, requires gcc/g++ >= 12.3")
        endif()
    else()
        message(WARNING "Disable AVX512-BF16 ISA support, no avx512_bf16 found in local CPU flags.")
    endif()
    
elseif (AVX2_FOUND)
    list(APPEND AIBRIX_CPU_FLAGS "-mavx2")
    message(WARNING "Using AVX2 ISA")
    
elseif (POWER9_FOUND OR POWER10_FOUND OR POWER11_FOUND)
    message(STATUS "PowerPC detected")
    if (POWER9_FOUND)
        list(APPEND AIBRIX_CPU_FLAGS
            "-mvsx"
            "-mcpu=power9"
            "-mtune=power9")
    elseif (POWER10_FOUND OR POWER11_FOUND)
        list(APPEND AIBRIX_CPU_FLAGS
            "-mvsx"
            "-mcpu=power10"
            "-mtune=power10")
    endif()

elseif (ASIMD_FOUND)
    message(STATUS "ARMv8 or later architecture detected")
    if(ARM_BF16_FOUND)
        message(STATUS "BF16 extension detected")
        set(MARCH_FLAGS "-march=armv8.2-a+bf16+dotprod+fp16")
        add_compile_definitions(ARM_BF16_SUPPORT)
    else()
        message(WARNING "BF16 functionality is not available")
        set(MARCH_FLAGS "-march=armv8.2-a+dotprod+fp16")  
    endif()
    list(APPEND AIBRIX_CPU_FLAGS ${MARCH_FLAGS})     
elseif(APPLE_SILICON_FOUND)
    message(STATUS "Apple Silicon Detected")
    set(ENABLE_NUMA OFF)
elseif (S390_FOUND)
    message(STATUS "S390 detected")
    # Check for S390 VXE support
    list(APPEND AIBRIX_CPU_FLAGS
        "-mvx"
        "-mzvector"
        "-march=native"
        "-mtune=native")
endif()

#
# Build oneTBB
#
FetchContent_Declare(
    oneTBB
    GIT_REPOSITORY https://github.com/uxlfoundation/oneTBB.git
    GIT_TAG v2022.2.0
    GIT_PROGRESS TRUE
    GIT_SHALLOW TRUE
)

set(TBB_TEST "OFF")
set(TBB_STRICT "OFF")
set(TBB_INSTALL "OFF")
set(BUILD_SHARED_LIBS "OFF")
set(TBB_BUILD "ON")
set(TBBMALLOC_BUILD "ON")
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

FetchContent_MakeAvailable(oneTBB)

list(APPEND LIBS TBB::tbb)
list(APPEND LIBS TBB::tbbmalloc)

if(ENABLE_NUMA)
    list(APPEND LIBS numa)
else()
    message(STATUS "NUMA is disabled")
    add_compile_definitions(-DAIBRIX_NUMA_DISABLED)
endif()

message(STATUS "CPU extension compile flags: ${AIBRIX_CPU_FLAGS}")

#
# _cpu_C extension
#
set(AIBRIX_CPU_EXT_SRC
    "csrc/cpu/tensor_pool.cpp"
    "csrc/cpu/torch_bindings.cpp")

#
# Define extension targets
#

define_gpu_extension_target(
    _cpu_C
    DESTINATION aibrix_kvcache
    LANGUAGE CXX
    SOURCES ${AIBRIX_CPU_EXT_SRC}
    LIBRARIES ${LIBS}
    COMPILE_FLAGS ${AIBRIX_CPU_FLAGS}
    INCLUDE_DIRECTORIES ${oneTBB_INCLUDES}
    USE_SABI 3
    WITH_SOABI)
