cmake_minimum_required(VERSION 3.12)

set(CUDA_SDK_ROOT_DIR "/usr/local/cuda/samples")

project(car_hyp_despot CXX CUDA)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED True)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED True)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -mfpmath=sse")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_30")
set(CMAKE_MODULE_PATH ${CMAKE_PREFIX_PATH} "${PROJECT_SOURCE_DIR}/cmake")

# GRPC
include(FetchContent)
set(ABSL_ENABLE_INSTALL ON)
FetchContent_Declare(
        gRPC
        GIT_REPOSITORY https://github.com/grpc/grpc
        GIT_TAG        v1.55.0  # e.g v1.28.0
)
set(FETCHCONTENT_QUIET OFF)
FetchContent_MakeAvailable(gRPC)
set(_PROTOBUF_LIBPROTOBUF libprotobuf)
set(_REFLECTION grpc++_reflection)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protoc>)
set(_GRPC_GRPCPP grpc++)
if(CMAKE_CROSSCOMPILING)
    find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
else()
    set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)
endif()

# GRPC


set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_30,code=sm_30
    )

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
  msg_builder
)

include_directories(
   ${catkin_INCLUDE_DIRS}
   src/gamma/
   src/HypDespot/include/
   src/HypDespot/include/despot/
   src/planner/
   ../crowd_pomdp_planner/src/
)
 
FILE(GLOB GAMMA_SRCS src/gamma/*.cpp)

catkin_package(
  INCLUDE_DIRS src/planner/
  LIBRARIES car_hyp_despot
  CATKIN_DEPENDS roscpp msg_builder 
  # DEPENDS CUDA
)

add_library(
"${PROJECT_NAME}" 
STATIC

  src/HypDespot/src/interface/pomdp.cpp
  src/HypDespot/src/interface/world.cpp
  src/HypDespot/src/interface/belief.cpp
  src/HypDespot/src/interface/lower_bound.cpp
  src/HypDespot/src/interface/upper_bound.cpp
  src/HypDespot/src/interface/default_policy.cpp
  src/HypDespot/src/interface/policy_graph.cpp
  src/HypDespot/src/core/particle_belief.cpp
  src/HypDespot/src/core/globals.cpp
  src/HypDespot/src/core/builtin_lower_bounds.cpp
  src/HypDespot/src/core/mdp.cpp
  src/HypDespot/src/core/node.cpp
  src/HypDespot/src/core/builtin_policy.cpp
  src/HypDespot/src/core/builtin_policygraph.cpp
  src/HypDespot/src/core/pomdp_world.cpp
  src/HypDespot/src/core/solver.cpp
  src/HypDespot/src/core/builtin_upper_bounds.cpp
  src/HypDespot/src/logger.cpp
  src/HypDespot/src/planner.cpp
  src/HypDespot/src/Parallel_planner.cu
  src/HypDespot/src/evaluator.cpp
  src/HypDespot/src/pomdpx/parser/function.cpp
  src/HypDespot/src/pomdpx/parser/parser.cpp
  src/HypDespot/src/pomdpx/parser/variable.cpp
  src/HypDespot/src/pomdpx/pomdpx.cpp
  src/HypDespot/src/random_streams.cpp
  src/HypDespot/src/plannerbase.cpp
  src/HypDespot/src/solver/aems.cpp
  src/HypDespot/src/solver/despot.cpp
  src/HypDespot/src/solver/pomcp.cpp
  src/HypDespot/src/solver/baseline_solver.cpp
  src/HypDespot/src/util/coord.cpp
  src/HypDespot/src/util/dirichlet.cpp
  src/HypDespot/src/util/exec_tracker.cpp
  src/HypDespot/src/util/floor.cpp
  src/HypDespot/src/util/gamma.cpp
  src/HypDespot/src/util/logging.cpp
  src/HypDespot/src/util/random.cpp
  src/HypDespot/src/util/seeds.cpp
  src/HypDespot/src/util/util.cpp
  src/HypDespot/src/util/error_handler.cpp
  src/HypDespot/src/util/tinyxml/tinystr.cpp
  src/HypDespot/src/util/tinyxml/tinyxml.cpp
  src/HypDespot/src/util/tinyxml/tinyxmlerror.cpp
  src/HypDespot/src/util/tinyxml/tinyxmlparser.cpp

  src/HypDespot/src/GPUinterface/GPUdefault_policy.cu  
  src/HypDespot/src/GPUinterface/GPUpomdp.cu 
  src/HypDespot/src/GPUinterface/GPUlower_bound.cu  
  src/HypDespot/src/GPUinterface/GPUupper_bound.cu  
  src/HypDespot/src/GPUcore/GPUbuiltin_lower_bound.cu
  src/HypDespot/src/GPUcore/GPUbuiltin_policy.cu
  src/HypDespot/src/GPUcore/GPUpolicy_graph.cu
  src/HypDespot/src/GPUcore/GPUbuiltin_upper_bound.cu
  src/HypDespot/src/GPUcore/GPUconfig.cu   
  src/HypDespot/src/GPUcore/GPUhistory.cu      
  src/HypDespot/src/GPUcore/GPUnode.cu  
  src/HypDespot/src/GPUcore/GPUglobals.cu  
  src/HypDespot/src/GPUcore/shared_node.cu
  src/HypDespot/src/GPUcore/thread_globals.cu
  src/HypDespot/src/GPUutil/GPUmemorypool.cu  
  src/HypDespot/src/GPUutil/GPUseeds.cu    
  src/HypDespot/src/GPUutil/GPUutil.cu
  src/HypDespot/src/GPUutil/GPUcoord.cu      
  src/HypDespot/src/GPUutil/GPUrandom.cu      
  src/HypDespot/src/GPUrandom_streams.cu
  src/HypDespot/src/solver/Hyp_despot.cu

  ${GAMMA_SRCS}

  src/planner/path.cpp
  src/planner/collision.cpp
  src/planner/context_pomdp.cpp
  src/planner/default_prior.cpp
  src/planner/world_model.cpp
  src/planner/crowd_belief.cpp  
  src/planner/param.cpp
  src/planner/moped_prediction.cpp
)

add_dependencies("${PROJECT_NAME}" ${catkin_EXPORTED_TARGETS})

set_target_properties( "${PROJECT_NAME}"
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties( "${PROJECT_NAME}"
                       PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(${PROJECT_NAME}
  PRIVATE KW_EXPORTS
  INTERFACE KW_IMPORTS
)

target_link_libraries("${PROJECT_NAME}"
  ${TinyXML_LIBRARIES}
)

## GRPC
# GRPC

set(_PROTOBUF_LIBPROTOBUF libprotobuf)
set(_REFLECTION grpc++_reflection)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protoc>)
set(_GRPC_GRPCPP grpc++)
if(CMAKE_CROSSCOMPILING)
    find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
else()
    set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)
endif()

# Proto file
get_filename_component(hw_proto "../crowd_pomdp_planner/protos/agentinfo.proto" ABSOLUTE)
get_filename_component(hw_proto_path "${hw_proto}" PATH)

# Generated sources
set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/agentinfo.pb.cc")
set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/agentinfo.pb.h")
set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/agentinfo.grpc.pb.cc")
set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/agentinfo.grpc.pb.h")

add_custom_command(
        OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}"
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${hw_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${hw_proto}"
        DEPENDS "${hw_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# hw_grpc_proto
add_library(hw_grpc_proto
        ${hw_grpc_srcs}
        ${hw_grpc_hdrs}
        ${hw_proto_srcs}
        ${hw_proto_hdrs})

target_link_libraries(hw_grpc_proto
        ${_REFLECTION}
        ${_GRPC_GRPCPP}
        ${_PROTOBUF_LIBPROTOBUF})

# Targets greeter_[async_](client|server)
target_link_libraries(${PROJECT_NAME}
        hw_grpc_proto
        ${_REFLECTION}
        ${_GRPC_GRPCPP}
        ${_PROTOBUF_LIBPROTOBUF})

