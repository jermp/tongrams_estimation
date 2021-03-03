cmake_minimum_required(VERSION 2.8)
project(TONGRAMS_ESTIMATION)

if(CMAKE_BUILD_TYPE MATCHES Debug)
  MESSAGE(STATUS "DEBUG defined")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
endif()

if(LSD_RADIX_SORT)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLSD_RADIX_SORT")
  MESSAGE(STATUS "Sorting with LSD_RADIX_SORT")
endif()

MESSAGE(STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})


if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif ()

if (UNIX AND NOT APPLE)
  MESSAGE(STATUS "Compiling with openmp")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp") # for __gnu_parallel::sort
endif()

if (UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-missing-braces")

  if(TONGRAMS_USE_SANITIZERS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
  endif()

endif()

find_package(Boost COMPONENTS iostreams filesystem thread REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

include_directories(${TONGRAMS_ESTIMATION_SOURCE_DIR}/include)
include_directories(${TONGRAMS_ESTIMATION_SOURCE_DIR}/external/tongrams/include)

add_subdirectory(external/tongrams)

file(GLOB SRC_SOURCES src/*.cpp)
foreach(SRC ${SRC_SOURCES})
  get_filename_component (SRC_NAME ${SRC} NAME_WE) # without extension
  add_executable(${SRC_NAME} ${SRC})
  target_link_libraries(${SRC_NAME} ${Boost_LIBRARIES})
endforeach(SRC)
