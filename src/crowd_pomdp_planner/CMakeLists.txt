cmake_minimum_required(VERSION 3.14.1)

set(CUDA_SDK_ROOT_DIR "/usr/local/cuda/samples")

project(crowd_pomdp_planner CXX CUDA)

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

#add_definitions(-std=c++0x -O3)

include(FetchContent)
set(ABSL_ENABLE_INSTALL ON)
FetchContent_Declare(
        gRPC
        GIT_REPOSITORY https://github.com/grpc/grpc
        GIT_TAG        v1.55.0  # e.g v1.28.0
)
#set(FETCHCONTENT_QUIET OFF)
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

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  std_msgs
  sensor_msgs
  nav_msgs
  msg_builder
  tf
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs msg_builder car_hyp_despot  roscpp
  # DEPENDS CUDA
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ../car_hyp_despot/src/gamma/
  ../car_hyp_despot/src/planner/
  ../car_hyp_despot/src/HypDespot/include/
  ../car_hyp_despot/src/HypDespot/include/despot
  ../crowd_pomdp_planner/src/
)

add_library(car_hyp_despot STATIC IMPORTED)
set_target_properties(car_hyp_despot PROPERTIES
  IMPORTED_LOCATION ${CATKIN_DEVEL_PREFIX}/lib/libcar_hyp_despot.a 
)
set_target_properties(car_hyp_despot PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES "CUDA"
)

add_executable(ped_pomdp src/context_pomdp_node.cpp src/controller.cpp src/world_simulator.cpp)
add_executable(local_frame src/local_frame.cpp)
add_executable(vel_publisher src/vel_publisher.cpp)

set_target_properties( ped_pomdp 
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties( ped_pomdp
                       PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(ped_pomdp
  car_hyp_despot
  ${catkin_LIBRARIES}
)

target_link_libraries(local_frame
  car_hyp_despot
  ${catkin_LIBRARIES}
)

target_link_libraries(vel_publisher
  car_hyp_despot
  ${catkin_LIBRARIES}
)

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
get_filename_component(hw_proto "./protos/agentinfo.proto" ABSOLUTE)
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
target_link_libraries(ped_pomdp
        hw_grpc_proto
        ${_REFLECTION}
        ${_GRPC_GRPCPP}
        ${_PROTOBUF_LIBPROTOBUF})
