cmake_minimum_required(VERSION 2.8)
project(gPb)

find_package(OpenCV)

# Deal with gpb_src
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
include_directories(gpb_src/include buildW/util)
add_library(opencv_gpb SHARED
                              gpb_src/src/concurrent/threads/auto_parallelize.cc
                              gpb_src/src/concurrent/threads/child_thread.cc
                              gpb_src/src/concurrent/threads/runnable.cc
                              gpb_src/src/concurrent/threads/thread.cc
                              gpb_src/src/concurrent/threads/synchronization/synchronizables/unsynchronized.cc
                              gpb_src/src/concurrent/threads/synchronization/synchronizables/synchronized.cc
                              gpb_src/src/concurrent/threads/synchronization/synchronizables/synchronizable.cc
                              gpb_src/src/lang/array.cc
                              gpb_src/src/lang/exceptions/ex_index_out_of_bounds.cc
                              gpb_src/src/lang/exceptions/exception.cc
                              gpb_src/src/lang/exceptions/ex_bad_cast.cc
                              gpb_src/src/lang/exceptions/ex_invalid_argument.cc
                              gpb_src/src/lang/exceptions/ex_not_found.cc
                              gpb_src/src/lang/exceptions/ex_not_implemented.cc
                              gpb_src/src/lang/exceptions/ex_null_pointer_dereference.cc
                              gpb_src/src/lang/exceptions/throwable.cc
                              gpb_src/src/math/exact.cc
                              gpb_src/src/math/libraries/lib_image.cc
                              gpb_src/src/math/libraries/lib_signal.cc
                              gpb_src/src/math/geometry/point_2D.cc
                              gpb_src/src/math/geometry/seg_intersect.cc
                              gpb_src/src/math/geometry/triangle_2D.cc
                              gpb_src/src/math/geometry/triangulation.cc
                              gpb_src/src/math/math.cc
                              gpb_src/src/math/matrices/exceptions/ex_matrix_dimension_mismatch.cc
                              gpb_src/src/math/matrices/matrix.cc
                              gpb_src/src/math/random/generators/rand_gen_uniform.cc
                              gpb_src/src/math/random/util/randperm.cc
                              gpb_src/src/math/random/sources/rand_source_default.cc
                              gpb_src/src/math/random/sources/mersenne_twister_64.cc
                              gpb_src/src/math/random/sources/rand_source.cc
                              gpb_src/src/math/random/sources/rand_source_64.cc
                              gpb_src/src/math/random/sources/system_entropy.cc
                              gpb_src/src/mlearning/clustering/clusterers/abstract/clusterer.cc
                              gpb_src/src/mlearning/clustering/clusterers/abstract/weighted_clusterer.cc
                              gpb_src/src/mlearning/clustering/clusterers/kmeans/basic_clusterer.cc

                              buildW/affinity.cc
                              buildW/ic.cc
                              buildW/smatrix.cc

                              opencv_gpb/src/watershed.cpp
)

target_link_libraries(opencv_gpb ${OpenCV_LIBRARIES})


# deal with buildW
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
include_directories(buildW/util)
add_library(buildW SHARED buildW/affinity.cc
                   buildW/ic.cc
                   buildW/smatrix.cc
)

target_link_libraries(buildW ${OpenCV_LIBRARIES})
